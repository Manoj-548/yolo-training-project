# TODO: Fix YOLO Weights Loading Error

- [x] Add import for DetectionModel from ultralytics.nn.tasks
- [x] Modify load_yolo_model function to add torch.serialization.add_safe_globals([DetectionModel]) before YOLO(w)
- [x] Modify validate_args function to add torch.serialization.add_safe_globals([DetectionModel]) before YOLO(args.weights)
- [x] Modify main function to add torch.serialization.add_safe_globals([DetectionModel]) before YOLO(base_weights)
