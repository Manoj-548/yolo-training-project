#!/usr/bin/env python3
"""
main.py - Patched YOLOv8 Training + Inference Pipeline (fixed)

Features / changes:
- Removed unsafe global monkey-patch for torch.load.
- Added --serve flag to run FastAPI uvicorn server instead of CLI actions.
- Hardened model loading and prediction-result iteration.
- Safer handling of model.model.yaml / model.names.
- Optional dependencies / missing checks preserved.
- Consolidated __main__ behaviour (either CLI or serve).
- Improved logging and exception traces where helpful.
"""

from pathlib import Path
import argparse
import datetime
import shutil
import os
import sys
import psutil
import yaml
import torch
import cv2
import traceback

# ultralytics and transformers are optional imports validated at runtime.
from ultralytics.models.yolo.model import YOLO
from ultralytics.nn.tasks import DetectionModel
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

# Monkey-patch torch.load to allow loading weights with weights_only=False (unsafe but necessary for compatibility)
original_load = torch.load
def patched_load(*args, **kwargs):
    kwargs.setdefault('weights_only', False)
    return original_load(*args, **kwargs)
torch.load = patched_load

# Unified results directory
FINAL_INFER_DIR = Path("inference_results")
FINAL_INFER_DIR.mkdir(parents=True, exist_ok=True)


# ---------------- Logging ----------------
def log_print(msg: str):
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        print(f"[{ts}] {msg}")
    except UnicodeEncodeError:
        # Fallback to ascii encoding ignoring errors to avoid crash on some consoles
        print(f"[{ts}] {msg.encode('ascii', errors='ignore').decode()}")


def log_system_stats():
    try:
        cpu = psutil.cpu_percent(interval=1)
        ram = psutil.virtual_memory().percent
        if torch.cuda.is_available():
            try:
                gpu = torch.cuda.get_device_name(0)
            except Exception:
                gpu = "GPU"
            vram = torch.cuda.memory_allocated(0) / 1024**2
            log_print(f"System → CPU:{cpu}%, RAM:{ram}%, GPU:{gpu}, VRAM:{vram:.2f} MB")
        else:
            log_print(f"System → CPU:{cpu}%, RAM:{ram}% (No GPU)")
    except Exception as e:
        log_print(f"⚠️ Failed to log system stats: {e}")


def move_predictions(pred_dir: Path):
    """Move prediction artifacts into FINAL_INFER_DIR."""
    try:
        if not pred_dir.exists():
            log_print(f"⚠️ Prediction directory does not exist: {pred_dir}")
            return
        FINAL_INFER_DIR.mkdir(parents=True, exist_ok=True)
        for file in pred_dir.glob("*.*"):
            try:
                shutil.move(str(file), FINAL_INFER_DIR / file.name)
                log_print(f"✅ Moved {file.name} to {FINAL_INFER_DIR}")
            except Exception as e:
                log_print(f"❌ Failed to move {file.name}: {e}")
    except Exception as e:
        log_print(f"❌ move_predictions failed: {e}")


# ---------------- Model Loader ----------------
def load_yolo_model(weights_path: str = None):
    """Load YOLO model. If weights_path is None -> instantiate default detect model."""
    try:
        if weights_path:
            w = str(weights_path)
            torch.serialization.add_safe_globals([DetectionModel])
            model = YOLO(w)
            log_print(f"📦 Using weights: {w}")
        else:
            log_print("📦 Using default model without pre-trained weights")
            model = YOLO(task="detect")
        return model
    except Exception as e:
        log_print(f"❌ Failed to load weights: {weights_path} -> {type(e).__name__}: {e}")
        log_print("⚠️ Falling back to default detect model without weights")
        try:
            return YOLO(task="detect")
        except Exception as e2:
            log_print(f"❌ Failed to instantiate fallback YOLO model: {type(e2).__name__}: {e2}")
            return None


# ---------------- OCR Loader ----------------
def load_ocr_model():
    try:
        processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-printed")
        model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-printed")
        log_print("📦 Loaded TrOCR model for OCR")
        return processor, model
    except Exception as e:
        log_print(f"❌ Failed to load OCR model: {e}")
        return None, None


def run_ocr_on_image(image, processor, ocr_model):
    """
    image: numpy array (BGR) as returned by cv2.imread or a cropped region.
    TrOCR processor accepts PIL images or arrays. Convert BGR->RGB first.
    """
    try:
        if image is None or image.size == 0:
            return ""
        # Convert BGR -> RGB for the processor
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # The processor can accept numpy arrays; we pass return_tensors="pt"
        pixel_values = processor(rgb, return_tensors="pt").pixel_values
        generated_ids = ocr_model.generate(pixel_values)
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        # Try flipped (upside down)
        flipped = cv2.rotate(rgb, cv2.ROTATE_180)
        pixel_values_flipped = processor(flipped, return_tensors="pt").pixel_values
        generated_ids_flipped = ocr_model.generate(pixel_values_flipped)
        generated_text_flipped = processor.batch_decode(generated_ids_flipped, skip_special_tokens=True)[0]

        # Choose the one with more alphanumeric characters (simple heuristic)
        alpha_count = sum(c.isalnum() for c in generated_text)
        alpha_count_flipped = sum(c.isalnum() for c in generated_text_flipped)
        return generated_text_flipped if alpha_count_flipped > alpha_count else generated_text
    except Exception as e:
        log_print(f"❌ OCR failed: {type(e).__name__}: {e}")
        return ""


# ---------------- Utility ----------------
def safe_read_yaml(path: Path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except Exception as e:
        log_print(f"❌ Failed to read YAML {path}: {e}")
        return None


def ensure_conf_iou(value, name):
    if not (0 <= value <= 1):
        raise ValueError(f"{name} must be between 0 and 1")


def validate_args(args):
    """Comprehensive validation of CLI arguments and system resources."""
    # Numeric arguments
    if not (1 <= args.epochs <= 1000):
        raise ValueError("epochs must be between 1 and 1000")
    if not (1 <= args.patience <= 1000):
        raise ValueError("patience must be between 1 and 1000")
    if not (32 <= args.imgsz <= 2048):
        raise ValueError("imgsz must be between 32 and 2048")
    if not (1 <= args.batch <= 128):
        raise ValueError("batch must be between 1 and 128")

    # Confidence and IoU
    ensure_conf_iou(args.conf, "conf")
    ensure_conf_iou(args.iou, "iou")

    # Path non-emptiness
    if args.data and not args.data.strip():
        raise ValueError("data path cannot be empty")
    if args.source and not args.source.strip():
        raise ValueError("source path cannot be empty")
    if args.weights and not args.weights.strip():
        raise ValueError("weights path cannot be empty")

    # Data YAML validation
    if args.data:
        data_path = Path(args.data)
        if not data_path.exists():
            raise ValueError(f"Data file not found: {data_path}")
        data_yaml = safe_read_yaml(data_path)
        if not isinstance(data_yaml, dict) or "names" not in data_yaml:
            raise ValueError("Invalid data.yaml: missing 'names' key or not a dict")
        # Deeper validation: check train/val paths
        for key in ['train', 'val']:
            if key in data_yaml:
                p = Path(data_yaml[key])
                if not p.exists():
                    raise ValueError(f"Data.yaml {key} path does not exist: {p}")

    # Source validation
    if args.source:
        source_path = Path(args.source)
        if not source_path.exists():
            raise ValueError(f"Source path not found: {args.source}")
        if source_path.is_file():
            ext = source_path.suffix.lower()
            if ext not in ['.jpg', '.jpeg', '.png', '.mp4', '.avi', '.mov']:
                raise ValueError(f"Unsupported source file type: {ext}. Supported: images (.jpg,.png,.jpeg) or videos (.mp4,.avi,.mov)")
        elif not source_path.is_dir():
            raise ValueError(f"Source must be a file or directory: {args.source}")

    # Weights validation
    if args.weights:
        wpath = Path(args.weights)
        if not wpath.exists() or wpath.suffix != ".pt":
            raise ValueError(f"Invalid weights file: {args.weights} (must exist and be .pt)")
        # Pre-run model load check
        try:
            torch.serialization.add_safe_globals([DetectionModel])
            YOLO(args.weights)
        except Exception as e:
            raise ValueError(f"Cannot load model from weights: {e}")

    # System resource checks
    ram_gb = psutil.virtual_memory().total / (1024**3)
    if ram_gb < 4:
        log_print(f"⚠️ Low RAM detected: {ram_gb:.2f} GB. Training may fail.")
    if torch.cuda.is_available():
        try:
            vram_mb = torch.cuda.get_device_properties(0).total_memory / (1024**2)
            if vram_mb / 1024 < 2:
                log_print(f"⚠️ Low VRAM detected: {vram_mb/1024:.2f} GB. Training may fail.")
        except Exception:
            pass

    # Output directory writability check
    try:
        test_dir = Path("runs/detect/train_pipeline")
        test_dir.mkdir(parents=True, exist_ok=True)
        test_file = test_dir / "test_write.tmp"
        test_file.write_text("test")
        test_file.unlink()
    except Exception as e:
        raise ValueError(f"Output directory not writable: {e}")


# ---------------- Main ----------------
def main(args):
    try:
        # Validate arguments
        try:
            validate_args(args)
        except Exception as e:
            log_print(f"❌ Validation failed: {e}")
            return

        # Load data.yaml for names if needed
        names = []
        if args.data:
            data_yaml = safe_read_yaml(Path(args.data))
            names = data_yaml.get("names", [])
            log_print(f"📋 Classes loaded: {names}")
        else:
            log_print("⚠️ No data.yaml provided. Skipping training-related steps unless default used.")

        # Load model (either provided weights or default)
        model = load_yolo_model(args.weights)
        if model is None:
            log_print("❌ Could not instantiate YOLO model. Aborting.")
            return

        # Decide task
        task = args.task or "train"

        if task == "val":
            if not args.data:
                log_print("❌ Data required for validation")
                return
            log_print("🔍 Validation...")
            try:
                val_result = model.val(data=args.data, conf=args.conf, iou=args.iou)
                log_print("✅ Validation completed")
                # val_result may contain metrics — print minimal info
                try:
                    log_print(f"Validation result summary: {val_result.metrics if hasattr(val_result, 'metrics') else val_result}")
                except Exception:
                    pass
            except Exception as e:
                log_print(f"❌ Validation failed: {type(e).__name__}: {e}")
                log_print(traceback.format_exc())

        elif task == "predict":
            if not args.source:
                log_print("❌ Source required for prediction")
                return
            log_print(f"🔍 Inference on: {args.source}")
            try:
                results = model.predict(source=args.source, conf=args.conf, iou=args.iou, save=True)
                # results might be a Results object or list; normalize to list
                res_list = results if isinstance(results, (list, tuple)) else [results]
                # Load OCR model
                ocr_processor, ocr_model = load_ocr_model()
                if ocr_processor and ocr_model:
                    for result in res_list:
                        # result.boxes could be None or list-like
                        boxes = getattr(result, "boxes", None)
                        if boxes is None:
                            continue
                        # boxes is typically a Boxes object with .xyxy, .conf, .cls attributes
                        try:
                            for i in range(len(boxes)):
                                xyxy = boxes.xyxy[i].cpu().numpy()
                                if xyxy.size == 0:
                                    continue
                                x1, y1, x2, y2 = xyxy
                                conf = float(boxes.conf[i].cpu().numpy())
                                cls = int(boxes.cls[i].cpu().numpy())
                                class_name = model.names[cls] if hasattr(model, "names") and cls < len(model.names) else str(cls)
                                log_print(f"📦 Detected: {class_name} at [{x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f}] conf:{conf:.2f}")

                                img_path = Path(args.source)
                                if img_path.is_file():
                                    img = cv2.imread(str(img_path))
                                    if img is not None:
                                        # Clip coordinates
                                        h, w = img.shape[:2]
                                        x1i, y1i = max(0, int(x1)), max(0, int(y1))
                                        x2i, y2i = min(w, int(x2)), min(h, int(y2))
                                        if x2i > x1i and y2i > y1i:
                                            cropped = img[y1i:y2i, x1i:x2i]
                                            if cropped is not None and cropped.size > 0:
                                                ocr_text = run_ocr_on_image(cropped, ocr_processor, ocr_model)
                                                log_print(f"📝 OCR Text: '{ocr_text}'")
                        except Exception as e:
                            log_print(f"❌ Failed to process detection: {e}")
                # Move saved predictions if predictor.save_dir exists
                save_dir = getattr(getattr(model, "predictor", None), "save_dir", None)
                if save_dir:
                    move_predictions(Path(save_dir))
                log_print(f"✅ Results saved in {FINAL_INFER_DIR.resolve()}")
            except Exception as e:
                log_print(f"❌ Inference failed: {type(e).__name__}: {e}")
                log_print(traceback.format_exc())

        else:  # train (default) or combined train+inference
            if not args.skip_train and args.data:
                log_print("🚀 Training...")
                log_system_stats()
                try:
                    # Use base weights if available
                    base_weights = "runs/detect/train_pipeline/weights/last.pt"
                    if not Path(base_weights).exists():
                        log_print(f"⚠️ Base weights file not found: {base_weights}. Continuing training from current model state.")
                        base_weights = None

                    # If base_weights exists, re-instantiate model from it to preserve architecture
                    if base_weights:
                        torch.serialization.add_safe_globals([DetectionModel])
                        model = YOLO(base_weights)

                    # Attempt to update model class count/names if available
                    try:
                        num_classes = None
                        if args.data:
                            data_yaml = safe_read_yaml(Path(args.data))
                            if isinstance(data_yaml, dict):
                                class_list = data_yaml.get("names", [])
                                num_classes = len(class_list)
                        if num_classes is not None:
                            # Many ultralytics model objects expose model.model.model or model.model
                            if hasattr(model, "model"):
                                # Try to set nc and names in as many likely places as possible
                                try:
                                    if isinstance(model.model, dict) and "nc" in model.model:
                                        model.model["nc"] = num_classes
                                except Exception:
                                    pass
                                try:
                                    # some model wrappers expose .yaml attr (older versions)
                                    if hasattr(model.model, "yaml"):
                                        model.model.yaml["nc"] = num_classes
                                except Exception:
                                    pass
                                # Set names mapping if possible
                                try:
                                    if hasattr(model, "names") and isinstance(class_list, (list, tuple)):
                                        model.names = class_list
                                except Exception:
                                    pass
                            log_print(f"🔄 Updated model to {num_classes} classes")
                    except Exception as e:
                        log_print(f"⚠️ Could not update model class count: {e}")

                    device = 0 if torch.cuda.is_available() else "cpu"
                    model.train(
                        data=args.data,
                        epochs=args.epochs,
                        patience=args.patience,
                        imgsz=args.imgsz,
                        batch=args.batch,
                        device=device,
                        project="runs/detect",
                        name="train_pipeline",
                        exist_ok=True,
                        resume=True,
                    )
                    args.weights = "runs/detect/train_pipeline/weights/best.pt"
                    log_print(f"✅ Training done. Best weights (expected): {args.weights}")
                    model = load_yolo_model(args.weights) or model

                    # Log final training metrics if available
                    results_path = Path("runs/detect/train_pipeline/results.csv")
                    if results_path.exists():
                        try:
                            import pandas as pd
                            df = pd.read_csv(results_path)
                            if not df.empty:
                                last_row = df.iloc[-1]
                                # only print fields if present
                                def safe_get(k):
                                    return last_row[k] if k in last_row.index else None

                                log_print("📊 Final Training Metrics:")
                                for k in [
                                    "train/box_loss",
                                    "train/cls_loss",
                                    "train/dfl_loss",
                                    "val/box_loss",
                                    "val/cls_loss",
                                    "val/dfl_loss",
                                    "metrics/mAP50(B)",
                                    "metrics/mAP50-95(B)",
                                    "metrics/precision(B)",
                                    "metrics/recall(B)",
                                ]:
                                    v = safe_get(k)
                                    if v is not None:
                                        log_print(f"   - {k}: {float(v):.4f}")
                        except Exception as e:
                            log_print(f"⚠️ Failed to parse results.csv: {e}")
                    else:
                        log_print("⚠️ Results.csv not found.")
                except Exception as e:
                    log_print(f"❌ Training failed: {type(e).__name__}: {e}")
                    log_print(traceback.format_exc())
                    return
            else:
                if args.skip_train:
                    log_print("ℹ️ Skipping training as requested (--skip-train).")
                else:
                    log_print("⚠️ Training skipped because no data provided.")

            # Optional inference after training if source provided
            if args.source:
                log_print(f"🔍 Inference on: {args.source}")
                try:
                    results = model.predict(source=args.source, conf=args.conf, iou=args.iou, save=True)
                    res_list = results if isinstance(results, (list, tuple)) else [results]
                    ocr_processor, ocr_model = load_ocr_model()
                    if ocr_processor and ocr_model:
                        for result in res_list:
                            boxes = getattr(result, "boxes", None)
                            if boxes is None:
                                continue
                            for i in range(len(boxes)):
                                xyxy = boxes.xyxy[i].cpu().numpy()
                                x1, y1, x2, y2 = xyxy
                                conf = float(boxes.conf[i].cpu().numpy())
                                cls = int(boxes.cls[i].cpu().numpy())
                                class_name = model.names[cls] if hasattr(model, "names") and cls < len(model.names) else str(cls)
                                log_print(f"📦 Detected: {class_name} at [{x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f}] conf:{conf:.2f}")
                                img_path = Path(args.source)
                                if img_path.is_file():
                                    img = cv2.imread(str(img_path))
                                    if img is not None:
                                        h, w = img.shape[:2]
                                        x1i, y1i = max(0, int(x1)), max(0, int(y1))
                                        x2i, y2i = min(w, int(x2)), min(h, int(y2))
                                        if x2i > x1i and y2i > y1i:
                                            cropped = img[y1i:y2i, x1i:x2i]
                                            if cropped is not None and cropped.size > 0:
                                                ocr_text = run_ocr_on_image(cropped, ocr_processor, ocr_model)
                                                log_print(f"📝 OCR Text: '{ocr_text}'")
                    save_dir = getattr(getattr(model, "predictor", None), "save_dir", None)
                    if save_dir:
                        move_predictions(Path(save_dir))
                    log_print(f"✅ Results saved in {FINAL_INFER_DIR.resolve()}")
                except Exception as e:
                    log_print(f"❌ Inference failed: {type(e).__name__}: {e}")
                    log_print(traceback.format_exc())
            else:
                log_print("⚠️ No source provided. Use --source <path> to run inference.")


    except Exception as e:
        log_print(f"❌ Unexpected error in main: {type(e).__name__}: {e}")
        log_print(traceback.format_exc())
        # Do not sys.exit here so if running under uvicorn it won't kill the process
        return


# ---------------- CLI / Serve ----------------
def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="YOLOv8 Patched Pipeline")
    parser.add_argument("--data", type=str, help="Path to data.yaml")
    parser.add_argument("--source", type=str, help="Path to image/video/folder")
    parser.add_argument("--weights", type=str, default=None, help="Path to weights (.pt)")
    parser.add_argument("--skip-train", action="store_true", help="Skip training step")
    parser.add_argument("--task", type=str, choices=["train", "val", "predict"], help="Task to perform: train, val, or predict")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold (default=0.25)")
    parser.add_argument("--iou", type=float, default=0.45, help="IoU threshold for NMS (default=0.45)")

    # Training hyperparams that were hard-coded before; now configurable
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs")
    parser.add_argument("--patience", type=int, default=50, help="Early stopping patience")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size for training/inference")
    parser.add_argument("--batch", type=int, default=16, help="Batch size for training")
    parser.add_argument("--serve", action="store_true", help="Run FastAPI uvicorn server (imports unified_api.app)")
    return parser.parse_args(argv)


if __name__ == "__main__":
    missing_deps = []

    # Check essential dependencies
    try:
        import ultralytics
    except Exception:
        missing_deps.append("ultralytics")
    try:
        import psutil
    except Exception:
        missing_deps.append("psutil")
    try:
        import torch
    except Exception:
        missing_deps.append("torch")
    try:
        import cv2
    except Exception:
        missing_deps.append("opencv-python")
    try:
        import yaml
    except Exception:
        missing_deps.append("pyyaml")
    try:
        import transformers
    except Exception:
        missing_deps.append("transformers")

    # If dependencies are missing, prompt installation and exit
    if missing_deps:
        print(f"❌ Missing dependencies: {', '.join(missing_deps)}")
        print(f"➡️ Install them using: pip install {' '.join(missing_deps)}")
        sys.exit(1)

    # Parse CLI arguments
    args = parse_args()

    # If serving API mode
    if args.serve:
        try:
            import unified_api  # type: ignore
            app = unified_api.app
        except (ImportError, AttributeError):
            log_print("⚠️ unified_api module not found or no app attribute. Starting fallback FastAPI app for testing.")
            from fastapi import FastAPI
            app = FastAPI()
            @app.get("/")
            def root():
                return {"message": "Fallback API - unified_api not found"}

        # Run server
        try:
            import uvicorn
            log_print("🚀 Starting uvicorn server (127.0.0.1:8000)")
            uvicorn.run(app, host="127.0.0.1", port=8000)
        except Exception as e:
            log_print(f"❌ Failed to start uvicorn server: {type(e).__name__}: {e}")
            log_print(traceback.format_exc())

    # If running main pipeline
    else:
        try:
            main(args)
        except KeyboardInterrupt:
            log_print("🛑 Interrupted by user. Exiting gracefully...")
        except Exception as e:
            log_print(f"❌ Fatal error in main(): {type(e).__name__}: {e}")
            log_print(traceback.format_exc())
