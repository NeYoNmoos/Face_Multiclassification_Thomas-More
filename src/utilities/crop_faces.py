import os
import cv2
import numpy as np
# from retinaface import RetinaFace
import gc
import pandas as pd

def preprocess_face(face_img, target_size=(224, 224)):
    try:
        if face_img is None or face_img.size == 0:
            return None

        # Convert to RGB if needed
        if len(face_img.shape) == 2:
            face_img = cv2.cvtColor(face_img, cv2.COLOR_GRAY2RGB)
        elif face_img.shape[2] == 4:
            face_img = cv2.cvtColor(face_img, cv2.COLOR_BGRA2RGB)
        elif face_img.shape[2] == 3:
            face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)

        # Resize while maintaining aspect ratio
        aspect_ratio = face_img.shape[1] / face_img.shape[0]
        if aspect_ratio > 1:
            new_width = target_size[0]
            new_height = int(new_width / aspect_ratio)
        else:
            new_height = target_size[1]
            new_width = int(new_height * aspect_ratio)

        resized = cv2.resize(face_img, (new_width, new_height), interpolation=cv2.INTER_AREA)

        # Create blank canvas
        final_img = np.zeros((target_size[0], target_size[1], 3), dtype=np.uint8)

        # Center the image
        y_offset = (target_size[0] - new_height) // 2
        x_offset = (target_size[1] - new_width) // 2
        final_img[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = resized

        return final_img

    except Exception as e:
        print(f"Error preprocessing face: {str(e)}")
        return None

def detect_faces(image):
    from retinaface import RetinaFace
    detections = RetinaFace.detect_faces(image)
    face_boxes = []
    if isinstance(detections, dict):
        for _, face_data in detections.items():
            x1, y1, x2, y2 = face_data['facial_area']
            face_boxes.append([x1, y1, x2 - x1, y2 - y1])

    return sorted(face_boxes, key=lambda x: x[0])  


def extract_and_save_faces(images, labels, output_folder, batch_size=50, target_size=(224, 224)):
    os.makedirs(output_folder, exist_ok=True)

    for batch_start in range(0, len(images), batch_size):
        batch_images = images[batch_start:batch_start + batch_size]
        batch_labels = labels[batch_start:batch_start + batch_size]

        for (filename, image), image_labels in zip(batch_images, batch_labels):
            print(f"Processing {filename}...")

            if image_labels == ["nothing"]:
                continue

            # Convert to RGB for detection
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            face_boxes = detect_faces(rgb_image)

            # Skip if face count does not match label count
            # if len(face_boxes) != len(image_labels):
            #     print(f"Skipping {filename} - Face count mismatch")
            #     continue

            # Process and save each face
            for i, (box, label) in enumerate(zip(face_boxes, image_labels)):
                x, y, w, h = box

                # Add margin to bounding box
                margin = int(max(w, h) * 0.2)
                x = max(0, x - margin)
                y = max(0, y - margin)
                w = min(w + 2 * margin, image.shape[1] - x)
                h = min(h + 2 * margin, image.shape[0] - y)

                # Extract face region
                face = image[y:y+h, x:x+w]

                # Preprocess face
                processed_face = preprocess_face(face, target_size)
                if processed_face is None:
                    print(f"Warning: Could not process face in {filename}")
                    continue

                # Save face image in label folder
                label_folder = os.path.join(output_folder, label.lower())
                os.makedirs(label_folder, exist_ok=True)

                face_filename = f"{os.path.splitext(filename)[0]}_face_{i}.jpg"
                face_path = os.path.join(label_folder, face_filename)
                cv2.imwrite(face_path, cv2.cvtColor(processed_face, cv2.COLOR_RGB2BGR))

def load_images(image_folder, label_map=None):
    images = []
    image_labels = []

    for filename in os.listdir(image_folder):
        img_path = os.path.join(image_folder, filename)
        img = cv2.imread(img_path)

        if img is not None:
            images.append((filename, img))
            if label_map:
                image_labels.append(label_map.get(filename, [])) 

    return images, image_labels if label_map else None


def create_extractions(train_image_folder, label_csv_path, output_folder):
    label_data = pd.read_csv(label_csv_path)
    label_data['label_name'] = label_data['label_name'].apply(eval) 
    label_map = dict(zip(label_data['image'].astype(str).str.zfill(4) + ".jpg", label_data['label_name']))

    train_images, train_labels = load_images(train_image_folder, label_map=label_map)

    extract_and_save_faces(train_images, labels=train_labels, output_folder=output_folder, batch_size=50)

def detect_faces_test(image):
    from retinaface import RetinaFace
    detections = RetinaFace.detect_faces(image, threshold=0.5)
    face_boxes = []
    if isinstance(detections, dict):
        for _, face_data in detections.items():
            x1, y1, x2, y2 = face_data['facial_area']
            confidence = face_data.get('score', 0)
            face_boxes.append({
                'box': [x1, y1, x2 - x1, y2 - y1],
                'confidence': confidence,
                'landmarks': face_data.get('landmarks', {})
            })

    # Sort by x-position (faces from left to right)
    return sorted(face_boxes, key=lambda x: x['box'][0])

def get_centered_crop(image, target_size=(224, 224)):
    height, width = image.shape[:2]

    crop_size = min(width, height)
    start_x = (width - crop_size) // 2
    start_y = (height - crop_size) // 2
    
    crop = image[start_y:start_y+crop_size, start_x:start_x+crop_size]
    return cv2.resize(crop, target_size)

def extract_and_save_test_faces(images, output_folder, batch_size=50, target_size=(224, 224)):
    os.makedirs(output_folder, exist_ok=True)
    
    stats = {
        'total_images': 0,
        'faces_detected': 0,
        'fallback_crops': 0
    }

    for batch_start in range(0, len(images), batch_size):
        batch_images = images[batch_start:batch_start + batch_size]

        for filename, image in batch_images:
            stats['total_images'] += 1
            print(f"Processing {filename}...")

            # Convert to RGB for detection
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            face_detections = detect_faces_test(rgb_image)

            if not face_detections:
                print(f"No faces detected in {filename}. Using centered crop...")
                stats['fallback_crops'] += 1
                
                # Get centered crop
                processed_face = get_centered_crop(image, target_size)
                face_filename = f"{os.path.splitext(filename)[0]}_face_0.jpg"
                face_path = os.path.join(output_folder, face_filename)
                cv2.imwrite(face_path, cv2.cvtColor(preprocess_face(processed_face, target_size), cv2.COLOR_RGB2BGR))
                continue

            stats['faces_detected'] += len(face_detections)
            
            # Process and save each face
            for i, detection in enumerate(face_detections):
                box = detection['box']
                x, y, w, h = box

                # Dynamic margin based on face size and confidence
                base_margin = max(w, h) * 0.2
                confidence = detection.get('confidence', 0.5)
                margin = int(base_margin * (1 + (1 - confidence)))
                
                # Add margin to bounding box
                x = max(0, x - margin)
                y = max(0, y - margin)
                w = min(w + 2 * margin, image.shape[1] - x)
                h = min(h + 2 * margin, image.shape[0] - y)

                # Extract face region
                face = image[y:y+h, x:x+w]

                # Preprocess face
                processed_face = preprocess_face(face, target_size)
                if processed_face is None:
                    print(f"Warning: Could not process face {i} in {filename}")
                    continue

                # Save face image
                face_filename = f"{os.path.splitext(filename)[0]}_face_{i}.jpg"
                face_path = os.path.join(output_folder, face_filename)
                cv2.imwrite(face_path, cv2.cvtColor(processed_face, cv2.COLOR_RGB2BGR))

    print("\nProcessing complete:")
    print(f"Total images processed: {stats['total_images']}")
    print(f"Total faces detected: {stats['faces_detected']}")
    print(f"Fallback crops created: {stats['fallback_crops']}")
    print(f"Average faces per image with detections: {stats['faces_detected']/(stats['total_images']-stats['fallback_crops']):.2f}")
    
def process_and_crop_faces(test_image_folder, test_output_folder):
    test_images, _ = load_images(test_image_folder)
    extract_and_save_test_faces(test_images, output_folder=test_output_folder, batch_size=50)
    
    