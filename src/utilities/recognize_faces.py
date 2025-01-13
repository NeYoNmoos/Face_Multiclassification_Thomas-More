import os
import numpy as np
from keras_facenet import FaceNet
import tensorflow as tf
import json
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def preprocess_image(image_path, target_size=(160, 160)):
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=target_size)
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array /= 255.0 
    return img_array


def load_test_data(test_folder, target_size=(160, 160)):
    test_images = []
    filenames = []
    for filename in sorted(os.listdir(test_folder)):
        img_path = os.path.join(test_folder, filename)
        img = preprocess_image(img_path, target_size)
        test_images.append(img)
        filenames.append(filename)
    return np.array(test_images), filenames



def predict_faces(test_folder, label_map_path, loaded_model, output_path, image_folder):
    with open(label_map_path, 'r') as json_file:
        label_map = json.load(json_file)
        
    test_images, test_filenames = load_test_data(test_folder)

    predictions = loaded_model.predict(test_images)
    predicted_labels = np.argmax(predictions, axis=1)

    label_map_reverse = {int(k): v for k, v in label_map.items()}
    predicted_names = [label_map_reverse[int(label)] for label in predicted_labels]

    create_submission(test_filenames, predicted_names, output_path, image_folder)

    return predicted_names

import os
import pandas as pd

def create_submission(filenames, predictions, output_path, image_folder):
    base_filenames = [filename.split('_face')[0] for filename in filenames]

    grouped_results = {}
    for base, prediction in zip(base_filenames, predictions):
        if base not in grouped_results:
            grouped_results[base] = []
        grouped_results[base].append(prediction)

    all_filenames = sorted(
        [os.path.splitext(filename)[0] for filename in os.listdir(image_folder) if filename.endswith('.jpg')]
    )

    submission_data = []
    for filename in all_filenames:
        if filename == "0456":
            continue
        if filename in grouped_results:
            label_name = ";".join(grouped_results[filename])
        else:
            label_name = "nothing"  
        submission_data.append({"image": filename, "label_name": label_name})

    submission_df = pd.DataFrame(submission_data)
    submission_df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")

import os
import pandas as pd

def create_submission(filenames, predictions, output_path, image_folder):
    base_filenames = [filename.split('_face')[0] for filename in filenames]

    grouped_results = {}
    for base, prediction in zip(base_filenames, predictions):
        if base not in grouped_results:
            grouped_results[base] = []
        grouped_results[base].append(prediction)

    all_filenames = sorted(
        [os.path.splitext(filename)[0] for filename in os.listdir(image_folder) if filename.endswith('.jpg')]
    )

    submission_data = []
    for filename in all_filenames:
        if filename in grouped_results:
            label_name = ";".join(grouped_results[filename])
        else:
            label_name = "nothing"  
        submission_data.append({"image": filename, "label_name": label_name})

    submission_df = pd.DataFrame(submission_data)
    submission_df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")

def create_submission_file(image_folder, output_path, test_filenames, predicted_names):
    create_submission(test_filenames, predicted_names, output_path, image_folder)
