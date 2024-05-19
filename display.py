import json
from PIL import Image
import cv2
import torchvision.transforms as T
from labels import coco_labels, yolov5_labels, target_size
import numpy as np
import os

#display ground truth image
def display_ground_truth_image(image_path):
    annotation_base_path = image_path.replace('/images/', '/annotations/')
    json_file_path = os.path.splitext(annotation_base_path)[0] + '.json'
    image = cv2.imread(image_path)

    # Load annotations from JSON file
    with open(json_file_path, 'r') as file:
        annotations = json.load(file)

    # Draw each bounding box
    for annotation in annotations:
        # Ensure there's a 'bbox' key in the dictionary
        if 'bbox' in annotation:
            x, y, w, h = map(int, annotation['bbox'])
            category_id = annotation['category_id']
            label = coco_labels.get(category_id, 'Unknown')  # Default to 'Unknown' if ID not in mapping

            top_left = (x, y)
            bottom_right = (x + w, y + h)
            cv2.rectangle(image, top_left, bottom_right, (0, 0, 255), 3)  # Red color in BGR and 3 px thick
            cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)  # Blue color in BGR and 1 px thick

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image_rgb)
    pil_image = pil_image.resize(target_size)
    return pil_image


# Function to preprocess image
def preprocess_image(image, target_size):
    transform = T.Compose([
        T.Resize(target_size),
        T.ToTensor(),
    ])
    return transform(image).unsqueeze(0)


# Function to display evaluation image
def display_eval_image(img, results, this_model):
    results = results[0]
    img = np.array(img)
    for i, bbox in enumerate(results['boxes']):
        if results['scores'][i] > 0.5:
            xmin, ymin, xmax, ymax = map(int, bbox)
            cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            label = results['labels'][i].item()
            if "YoloV5" in this_model:
                label = yolov5_labels[label]
            else:
                label = coco_labels[label]
            score = results['scores'][i].item()
            cv2.putText(img, f'{label} {score:.2f}', (xmin, ymin-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_RGB2RGBA))
    return pil_img

def display_boxes(img, results, model_name):
    results = results[0]
    for i, bbox in enumerate(results['boxes']):
        if results['scores'][i] > 0.5:  # Display only detections with score > 0.5
            xmin, ymin, xmax, ymax = map(int, bbox)
            cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)  # Draw rectangle with green color
            label = results['labels'][i].item()
            #label_id = results['labels'][i].item()  # Get the label ID
            #label = COCO_INSTANCE_CATEGORY_NAMES[label_id] 
            score = results['scores'][i].item()
            cv2.putText(img, f'{label} {score:.2f}', (xmin, ymin-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            return img