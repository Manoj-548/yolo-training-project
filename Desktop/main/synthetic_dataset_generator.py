# synthetic_dataset_generator.py
# Generate a YOLO-compatible synthetic OCR dataset

import os
import random
import yaml
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

# ---------------- CONFIG ----------------
OUTPUT_DIR = Path("synthetic_dataset")
IMG_SIZE = (128, 128)  # image size
NUM_SAMPLES = 200      # images per class
FONTS = [
    "C:/Windows/Fonts/arial.ttf",
    "C:/Windows/Fonts/times.ttf",
    "C:/Windows/Fonts/verdana.ttf"
]

# Characters: digits, uppercase, lowercase, symbols
CHARACTERS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz!@#$%^&*()-_=+[]{};:',.<>?/|\\"

# ---------------- SETUP ----------------
(OUTPUT_DIR / "images" / "train").mkdir(parents=True, exist_ok=True)
(OUTPUT_DIR / "images" / "val").mkdir(parents=True, exist_ok=True)
(OUTPUT_DIR / "labels" / "train").mkdir(parents=True, exist_ok=True)
(OUTPUT_DIR / "labels" / "val").mkdir(parents=True, exist_ok=True)

# ---------------- GENERATION ----------------
def generate_char_image(char, font_path, img_path, label_path, class_id):
    """Generate single character image + YOLO label"""
    img = Image.new("L", IMG_SIZE, color=255)  # grayscale white bg
    draw = ImageDraw.Draw(img)

    font_size = random.randint(40, 90)
    font = ImageFont.truetype(font_path, font_size)

    w, h = draw.textsize(char, font=font)
    pos = ((IMG_SIZE[0] - w) // 2, (IMG_SIZE[1] - h) // 2)

    draw.text(pos, char, fill=0, font=font)

    # Save image
    img.save(img_path)

    # YOLO normalized bbox: (class, x_center, y_center, width, height)
    x_center = (pos[0] + w / 2) / IMG_SIZE[0]
    y_center = (pos[1] + h / 2) / IMG_SIZE[1]
    bw = w / IMG_SIZE[0]
    bh = h / IMG_SIZE[1]

    with open(label_path, "w") as f:
        f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {bw:.6f} {bh:.6f}\n")

# ---------------- MAIN ----------------
def main():
    log = []
    for class_id, char in enumerate(CHARACTERS):
        for i in range(NUM_SAMPLES):
            subset = "train" if random.random() > 0.2 else "val"  # 80/20 split
            img_name = f"{char}_{i}.jpg"

            img_path = OUTPUT_DIR / "images" / subset / img_name
            label_path = OUTPUT_DIR / "labels" / subset / img_name.replace(".jpg", ".txt")

            generate_char_image(
                char,
                random.choice(FONTS),
                img_path,
                label_path,
                class_id
            )
        log.append(f"Class {class_id}: {char}")
    print("\n✅ Dataset generated in", OUTPUT_DIR)
    print("\n".join(log))

    # Create data.yaml
    yaml_path = OUTPUT_DIR / "data.yaml"
    data = {
        "train": str((OUTPUT_DIR / "images" / "train").resolve()),
        "val": str((OUTPUT_DIR / "images" / "val").resolve()),
        "nc": len(CHARACTERS),
        "names": list(CHARACTERS)
    }
    with open(yaml_path, "w") as f:
        yaml.dump(data, f)

    print(f"\n📄 data.yaml created at {yaml_path}")

if __name__ == "__main__":
    main()
