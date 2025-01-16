# Face Detection and Recegonitions Project - Thomas More AI-Frameworks

This project focuses on detecting and recognizing faces in images, specifically identifying classmates from a provided dataset. The pipeline involves data cleaning, preprocessing, face detection, and training a face recognition model to classify faces into one of 14 classes.

The project was part of the AI-Frameworks course at Thomas More University of Applied Sciences, supervised by Professor Daiane Saibert.

## Features

- Data Cleaning: Ensures consistent formatting for image and label data.
- Face Detection: Uses RetinaFace for robust face detection with fallback cropping for undetected faces
- Image Preprocessing: Normalizes and resizes images for training and testing
- Face Recognition: Implements and evaluates Fine-tuned Facenet model
- Kaggle Submissions: Outputs predictions for Kaggle competitions to evaluate performance

## File Structure

```
├── data_test/
│ ├── faces/                         # Cropped Faces
│ ├── test_images/              # Original and cleaned images
│ └── labels/                       # Label CSV files
├── models/                         # Saved models (.json and .h5)
├── report/                           # Comprehensive project report
├── src/                                # Source code for the project
│ ├── utilities/                     # Helper functions for preprocessing, detection, etc.
│ ├── 01_clean_label_data.ipynb
│ ├── 02_clean_image_data.ipynb
│ ├── 03_crop_faces_retinaface.ipynb
│ ├── 04_train_model_facenet_finetuning.ipynb
│ └── main.ipynb               # Test pipeline to preproocess, crop faces, load model and create predictions
├── requirements.txt           # Python dependencies
└── README.md              # Project documentation
```

## Requirements

Python 3.10.12

Install the required packages using the `requirements.txt` file:

```bash
pip install -r requirements.txt
```

## Usage

Load images into test_images/original_images folder.

Execute cells in main.ipynb file.
(second cell might not work -> follow instructions and execute cells in 03_crop_faces_deepface.ipynb instead)

## Documenation

A comprehensive report aboat all approaches used, aswell as evaluation and personal experiences can be found in report/ folder.

## Dependencies

This project uses the following Python libraries:

- `retina-face`
- `deepface`
- `keras`
- `keras-facenet`
- `keras`
- `mtcnn`
- `numpy`
- `opencv-python`
- `pandas`
- `pillow`
- `pillow_heif`
- `scikit-learn`
- `scipy`
- `split-folders`
- `tf-keras`
