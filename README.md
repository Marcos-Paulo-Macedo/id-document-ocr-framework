# 📸 Multi-Layer Computer Vision Framework for Document Classification & OCR

[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-Image_Processing-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org/)
[![Flask](https://img.shields.io/badge/Flask-Web_API-000000?style=for-the-badge&logo=flask&logoColor=white)](https://flask.palletsprojects.com/)

An end-to-end open-source Computer Vision and Deep Learning architecture designed to parse, classify, and extract structured text from complex, heterogeneous identity documents. 

This framework solves real-world OCR issues (glare, rotation, low contrast, and non-standard layouts) by combining adaptive image processing, transfer learning (EfficientNetB0), multi-layer voting consensus, and EasyOCR validation.

---

## 🚀 Key Features & Pipeline Architecture

1. **Automated Document Detection & Cropping:**
   - Detects document boundaries using `cv2.findContours` and 4-point polygon approximation to eliminate backgrounds.

2. **Adaptive Binarization & Preprocessing (v38 Engine):**
   - Applies local contrast enhancement using **CLAHE** (Contrast Limited Adaptive Histogram Equalization).
   - Dynamic noise reduction and adaptive Gaussian thresholding for high-contrast OCR ingestion.

3. **Multi-Layer Consensus Voting Engine:**
   - Combines predictions across multiple image variations (Adaptive B&W, Header Cropping, Canny Edge Detection, Binarized Otsu).
   - Executes a fallback mechanism linking heuristic keyword verification with CNN probability score thresholds.

4. **Deep Learning Transfer Learning Model:**
   - Powered by **EfficientNetB0** fine-tuned on custom document datasets (Two-phase training: Feature extraction + Deep layer fine-tuning).
   - Achieves 99.6% precision on heterogeneous validation subsets.

5. **Flask Audit & Re-Training Interface:**
   - Embedded Web UI for visual inspection, real-time threshold adjustment, and human-in-the-loop retraining data collection.

---

## 🛠 Tech Stack

- **Deep Learning:** TensorFlow 2.x, Keras, EfficientNetB0, MobileNetV2
- **Computer Vision:** OpenCV (`cv2`), NumPy, PIL
- **OCR Engine:** EasyOCR, PyTesseract
- **Web Interface:** Flask, HTML5, Tailwind CSS

---

## ⚙️ How to Run

### 1. Installation
```bash
git clone [https://github.com/Marcos-Paulo-Macedo/id-document-ocr-framework.git](https://github.com/Marcos-Paulo-Macedo/id-document-ocr-framework.git)
cd id-document-ocr-framework
pip install -r requirements.txt
