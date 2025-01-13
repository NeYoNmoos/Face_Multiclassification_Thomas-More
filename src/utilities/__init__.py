from .model_loading import load_facenet_model
from .clean_images import process_files_in_folder
from .crop_faces import create_extractions, process_and_crop_faces
from .recognize_faces import predict_faces

__version__ = "1.0.0"
__author__ = "Matthias Hefel"
