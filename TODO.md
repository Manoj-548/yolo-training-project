# TODO List for OCR Image Inspection and Model Enhancement

## 1. Create ocr_inspect.py

- [x] Inspect the OCR image (../../../../Downloads/ocr-a-font-sample.png) using default.yaml settings.
- [x] Extract text features: bounding boxes, confidence scores, and class indices.
- [x] Handle any number of characters dynamically.
- [x] Save extracted features to a JSON file (e.g., ocr_features.json).

## 2. Enhance main.py

- [ ] Add function to generate synthetic data similar to extracted features from ocr_inspect.py. (Currently generates blank images; need to create actual synthetic text images.)
- [x] Add function to train the model on the synthetic data using best.pt weights.
- [x] Ensure integration with existing training pipeline.

## 3. Modify infer_last.py

- [x] Load the trained model from main.py enhancements.
- [x] Use the model for predictions on the OCR image, incorporating results from main.py. (Currently hardcoded to OCR image; update for new images if needed.)

## 4. Testing and Validation

- [ ] Test ocr_inspect.py on the OCR image.
- [ ] Test synthetic data generation and training in main.py.
- [ ] Test predictions in infer_last.py with new images.
- [ ] Verify visual bounding boxes and confidence scores are displayed correctly.

## 5. Additional Tasks

- [ ] Copy trained weights: copy runs\detect\train_pipeline\weights\last.pt ../../../../Desktop/main/last.pt
- [ ] Generate proper synthetic data: Update generate_synthetic_data_from_features to create images with rendered text based on features.

## 6. New Feature: Inspect Any New Image from Dataset

- [ ] Create or enhance a script to inspect any new image from the company's project dataset (e.g., images/test/ directory).
- [ ] Extract insights and features from the images, similar to ocr_inspect.py.
- [ ] Allow dynamic inspection of multiple images, providing bounding boxes, confidence scores, class indices, and other relevant insights.
- [ ] Save extracted features to JSON or other formats for further analysis.

## 7. Add Segmentation for Character Detection

- [ ] Implement segmentation capabilities in the model training pipeline to better detect characters in images.
- [ ] Modify main.py or create a new script to train the model with segmentation tasks for character recognition.
- [ ] Update best.pt weights to include segmentation features for improved character detection.
- [ ] Test segmentation on the custom image (HT-GE134GC-T1-C-Snapshot-20250626-121953-875-886768416150.BMP) to ensure the model learns relevant features.
