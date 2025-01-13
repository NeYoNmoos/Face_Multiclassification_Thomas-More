from PIL import Image
from pillow_heif import register_heif_opener

register_heif_opener()

def convert_heic_to_jpg(heic_path, output_path):
    img = Image.open(heic_path)
    img = img.convert("RGB")  
    img.save(output_path, "JPEG")

import cv2
import os

def extract_frames(video_path, output_dir, frame_rate=1):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video file: {video_path}")
        return

    frame_idx = 0
    while True:
        success, frame = cap.read()
        if not success:
            break
        if frame_idx % frame_rate == 0:  
            output_path = os.path.join(output_dir, f"frame_{frame_idx}.jpg")
            cv2.imwrite(output_path, frame)
        frame_idx += 1
    cap.release()

from PIL import Image
from pillow_heif import register_heif_opener

register_heif_opener()

def process_files_in_folder(input_folder, output_folder, size=(224, 224)):
    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(input_folder):
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, os.path.splitext(filename)[0] + ".jpg")

        try:
            if filename.lower().endswith((".heic", ".heif")):
                img = Image.open(input_path)
                img = img.convert("RGB")  
                img = img.resize(size)  
                img.save(output_path, "JPEG")

            elif filename.lower().endswith((".jpg", ".jpeg", ".png")):
                img = cv2.imread(input_path)
                resized_img = cv2.resize(img, size)  
                cv2.imwrite(output_path, resized_img)

            elif filename.lower().endswith(".mp4"):
                extract_first_frame_from_video(input_path, output_path, size)

            else:
                print(f"Skipping unsupported file format: {filename}")
        except Exception as e:
            print(f"Error processing {filename}: {e}")

def extract_first_frame_from_video(video_path, output_path, size=(224, 224)):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video file: {video_path}")
        return

    success, frame = cap.read()
    if success:
        resized_frame = cv2.resize(frame, size)  
        cv2.imwrite(output_path, resized_frame) 
    else:
        print(f"Failed to read the first frame of video: {video_path}")

    cap.release()
