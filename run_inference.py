import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn, maskrcnn_resnet50_fpn, ssd300_vgg16, retinanet_resnet50_fpn
from effdet import get_efficientdet_config, EfficientDet, DetBenchTrain
from effdet.efficientdet import HeadNet
import cv2
from PIL import Image
import numpy as np
import requests
import torchvision.transforms as T

# COCO_INSTANCE_CATEGORY_NAMES = [
#     '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
#     'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
#     'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
#     'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
#     'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
#     'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 
#     'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
#     'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
#     'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'TV', 'laptop',
#     'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
#     'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
#     'hair drier', 'toothbrush'
# ]


# Function to load the entire model from a file
def load_entire_model(model_name):
    return torch.load(f'{model_name}.pth', map_location=torch.device('cpu'))

# Load models
yolov5_model = torch.hub.load(r'/Users/siravindran/programming/cv_performance/yolov5', 'custom', path=r'yolov5s.pt', source='local')
fasterrcnn_model = load_entire_model('fasterrcnn')
maskrcnn_model = load_entire_model('maskrcnn')
ssd_model = load_entire_model('ssd')
# effdet_model = load_entire_model('efficientdet')
retinanet_model = load_entire_model('retinanet')
print("All models loaded successfully.")

# Set models to evaluation mode
fasterrcnn_model.eval()
maskrcnn_model.eval()
ssd_model.eval()
#effdet_model.eval()
retinanet_model.eval()

# Define target size (resize to 512x512 for EfficientDet)
target_size = (512, 512)

# Load image
img_path = 'https://ultralytics.com/images/zidane.jpg'
img = Image.open(requests.get(img_path, stream=True).raw).convert("RGB")
img = img.resize(target_size)
cv_image = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
cv_image = cv2.resize(cv_image, target_size)

# Convert image to tensor
def preprocess_image(image, target_size):
    transform = T.Compose([
        T.Resize(target_size),
        T.ToTensor(),
    ])
    return transform(image).unsqueeze(0)


# Preprocess image
input_tensor = preprocess_image(img, target_size)

# Perform inference
with torch.no_grad():
    yolo_results = yolov5_model(img)
    fasterrcnn_results = fasterrcnn_model(input_tensor)
    maskrcnn_results = maskrcnn_model(input_tensor)
    ssd_results = ssd_model(input_tensor)
    retinanet_results = retinanet_model(input_tensor)

# # Function to convert YOLOv5 results to standard format
# def convert_yolo_results(yolo_results):
#     results = {
#         'boxes': [],
#         'labels': [],
#         'scores': []
#     }
#     for det in yolo_results.xyxy[0]:
#         xmin, ymin, xmax, ymax, conf, cls = det[:6]
#         results['boxes'].append([xmin.item(), ymin.item(), xmax.item(), ymax.item()])
#         results['labels'].append(int(cls.item()))
#         results['scores'].append(conf.item())
#     results_final = [results]
#     return results_final

# Function to convert YOLOv5 results to a format similar to Faster R-CNN
def convert_yolo_results(yolo_results):
    boxes = []
    labels = []
    scores = []

    # Extract information from each detection
    for det in yolo_results.xyxy[0]:
        xmin, ymin, xmax, ymax, conf, cls = det[:6]
        boxes.append([xmin.item(), ymin.item(), xmax.item(), ymax.item()])
        labels.append(int(cls.item()))
        scores.append(conf.item())

    # Convert lists to PyTorch tensors
    boxes_tensor = torch.tensor(boxes)
    labels_tensor = torch.tensor(labels)
    scores_tensor = torch.tensor(scores)

    # Create the final results dictionary in a list
    results_final = [{
        'boxes': boxes_tensor,
        'labels': labels_tensor,
        'scores': scores_tensor
    }]
    return results_final

# Convert YOLOv5 results
yolo_converted_results = convert_yolo_results(yolo_results)

# # Display results
print("YoloV5 results:")
print(yolo_converted_results)
print("Faster R-CNN results:")
print(fasterrcnn_results)
print("Mask R-CNN results:")
print(maskrcnn_results)
print("SSD results:")
print(ssd_results)
print("RetinaNet results:")
print(retinanet_results)

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
    cv2.imshow(model_name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Display results on the image
display_boxes(cv_image, yolo_converted_results,"SSD Results" )