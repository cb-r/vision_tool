import streamlit as st
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn, maskrcnn_resnet50_fpn, ssd300_vgg16, retinanet_resnet50_fpn
import cv2
from PIL import Image
import numpy as np
import pandas as pd
import os
from time import time
import plotly.express as px
import json
from utilities import system_info, timestamped_message
from display import preprocess_image, display_ground_truth_image, display_eval_image, display_boxes
from labels import target_size, coco_labels, yolov5_labels
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import plotly.figure_factory as ff


model_metrics = {
    'latency': {},
    'precision': {},
    'recall': {},
    'f1_score': {},
    'true_positive': {},
    'false_positive': {},
    'false_negative': {},
    'confusion_matrix': {},
    'image':{}
}


def load_ground_truth(image_path, original_size, target_size):
    annotation_base_path = image_path.replace('/images/', '/annotations/')
    json_file_path = os.path.splitext(annotation_base_path)[0] + '.json'
    with open(json_file_path, 'r') as file:
        annotations = json.load(file)

    ground_truth = {'boxes': [], 'labels': []}
    width_scale = target_size[0] / original_size[0]
    height_scale = target_size[1] / original_size[1]
    for annotation in annotations:
        bbox = annotation['bbox']
        category_id = annotation['category_id']
        bbox_converted = [
            bbox[0] * width_scale,
            bbox[1] * height_scale,
            (bbox[0] + bbox[2]) * width_scale,
            (bbox[1] + bbox[3]) * height_scale
        ]
        ground_truth['boxes'].append(bbox_converted)
        ground_truth['labels'].append(category_id)
    
    return ground_truth

# Function to convert YOLO results
def convert_yolo_results(yolo_results):
    boxes = []
    labels = []
    scores = []

    for det in yolo_results.xyxy[0]:
    #for det in yolo_results[0]:
        xmin, ymin, xmax, ymax, conf, cls = det[:6]
        boxes.append([xmin.item(), ymin.item(), xmax.item(), ymax.item()])
        labels.append(int(cls.item()))
        scores.append(conf.item())

    results_final = [{
        'boxes': torch.tensor(boxes),
        'labels': torch.tensor(labels),
        'scores': torch.tensor(scores)
    }]
    return results_final


# Function to evaluate model
def model_eval(img, model, selected_model):
    input_tensor = preprocess_image(img, target_size)
    with torch.no_grad():
        start_time = time()  
        if "YoloV5" in selected_model:
            yolo_results = model(img)
            cv_image = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            cv_image = cv2.resize(cv_image, target_size)
            converted_results = convert_yolo_results(yolo_results)
            final_img = display_boxes(cv_image, converted_results,"YOLO Results" )
        else:
            eval_results = model(input_tensor)
            final_img = img
            converted_results = eval_results
        eval_time = time() - start_time 
        return final_img, converted_results, eval_time


def calculate_iou(box1, box2):
    """Calculate intersection over union for two boxes."""
    # Coordinates of the intersection box
    
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    # Area of intersection
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    
    if intersection == 0:
        return 0.0

    # Areas of the individual boxes
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    # Union of both boxes
    union = box1_area + box2_area - intersection
    
    if union == 0:
        return 0.0
    
    # Intersection over union
    iou = intersection / union
    return iou


def calculate_metrics(ground_truth, predictions, this_model):
    gt_boxes = ground_truth['boxes']
    gt_labels = ground_truth['labels']
    pred_boxes = predictions['boxes'].numpy()  # Assuming the tensor is on CPU
    pred_labels = predictions['labels'].numpy()

    true_positives = 0
    false_positives = 0
    false_negatives = 0

    # Calculate IoUs and determine TP, FP, FN
    ious = []
    for i, gt_box in enumerate(gt_boxes):
        matched = False
        for j, pred_box in enumerate(pred_boxes):
            iou = calculate_iou(gt_box, pred_box)
            this_gt_label_string = coco_labels[gt_labels[i]]
            if "YoloV5" in this_model:
                this_pred_label_string = yolov5_labels[pred_labels[j]]
            else:
                this_pred_label_string = coco_labels[pred_labels[j]]
            if iou >= 0.1 and this_pred_label_string.lower() == this_gt_label_string.lower():
                true_positives += 1
                matched = True
                break
        if not matched:
            false_negatives += 1
    
    false_positives = len(pred_boxes) - true_positives

    # Calculating metrics
    precision = true_positives / (true_positives + false_positives) if true_positives + false_positives > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if true_positives + false_negatives > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0
    
    # num_classes = max(len(coco_labels), len(yolov5_labels))
    # confusion_matrix = np.zeros((num_classes, num_classes), dtype=int)
    # for i, gt_label in enumerate(gt_labels):
    #     if i < len(pred_labels):
    #         pred_label = pred_labels[i]
    #         if gt_label < num_classes and pred_label < num_classes:
    #             confusion_matrix[gt_label, pred_label] += 1
    confusion_matrix = 0
    return precision, recall, f1_score, iou, true_positives, false_positives, false_negatives, confusion_matrix


# Function to calculate metrics
def calc_metrics(selected_models, dataset_path):
    st.text(timestamped_message(f"Commenced performance metrics computations"))
    all_selected_models = []
    global model_metrics
    num_classes = len(coco_labels)
    model_metrics = {
        'latency': {model: [] for model in selected_models},
        'accuracy': {model: [] for model in selected_models},
        'precision': {model: [] for model in selected_models},
        'recall': {model: [] for model in selected_models},
        'f1_score': {model: [] for model in selected_models},
        'confusion_matrix': {model: np.zeros((num_classes, num_classes), dtype=int) for model in selected_models},
        'true_positive': {model: [] for model in selected_models},
        'false_positive': {model: [] for model in selected_models},
        'false_negative': {model: [] for model in selected_models},
        'image': {model: [] for model in selected_models}
}
   # Initialize session state for each model
    for model in selected_models:
        if model not in model_metrics['latency']:
            model_metrics['latency'][model] = []
    for this_model in selected_models:
        if "YoloV5" in this_model:
            model = torch.hub.load('yolov5', 'custom', path='yolov5s.pt', source='local')
        elif "Faster RCNN" in this_model:
            model = fasterrcnn_resnet50_fpn(pretrained=True)
        elif "Mask RCNN" in this_model:
            model = maskrcnn_resnet50_fpn(pretrained=True)
        elif "SSD" in this_model:
            model = ssd300_vgg16(pretrained=True)
        elif "RetinaNet" in this_model:
            model = retinanet_resnet50_fpn(pretrained=True)
        else:
            st.error(timestamped_message("Selected model is not supported."))
            continue
        model.eval()
        all_selected_models.append(model)

    st.text(timestamped_message(f"All selected models loaded successfully."))
    with st.expander("MODEL DETAILS"):
        st.write(all_selected_models)

    image_directory = os.path.join(dataset_path, 'images')
    image_files = [os.path.join(image_directory, filename) for filename in os.listdir(image_directory) if filename.lower().endswith(('png', 'jpg', 'jpeg'))]

    with st.expander("DATASET EVALUATIONS"):
        for image_path in image_files:
            img = Image.open(image_path).convert("RGB")
            original_size = img.size
            img = img.resize(target_size)
            cols = st.columns(len(selected_models) + 1)
            for idx, this_model in enumerate(selected_models):
                relative_path = image_path.replace(dataset_path+"/images/", "")
                with cols[idx]:
                    final_img, this_model_results, eval_time = model_eval(img, all_selected_models[idx], this_model)
                    ground_truth = load_ground_truth(image_path, original_size, target_size)  
                    # print ("\n"+this_model+"-"+image_path)
                    precision, recall, f1_score, iou, true_positives, false_positives, false_negatives, confusion_matrix = calculate_metrics(ground_truth, this_model_results[0], this_model)
                    st.image(display_eval_image(img, this_model_results, this_model), caption=f"{this_model},\nLatency : {eval_time:.2f} sec,Precision : {precision*100:.2f}%,Recall : {recall*100:.2f}%, F1 score: {f1_score}, IoU: {iou}, True positives: {true_positives}, False positives: {false_positives}, False Negatives: {false_negatives}")
                    # model_accuracies[this_model].append(accuracy)
                    # calculate_precision_recall_f1(ground_truth, this_model_results[0])
                    model_metrics['latency'][this_model].append(eval_time)
                    model_metrics['precision'][this_model].append(precision)
                    model_metrics['recall'][this_model].append(recall)
                    model_metrics['f1_score'][this_model].append(f1_score)
                    model_metrics['true_positive'][this_model].append(true_positives)
                    model_metrics['false_positive'][this_model].append(false_positives)
                    model_metrics['false_negative'][this_model].append(false_negatives)
                    model_metrics['confusion_matrix'][this_model] += confusion_matrix
                    model_metrics['image'][this_model].append(relative_path)

                    # model_metrics['confusion_matrix'][this_model].append(cm)

            with cols[-1]:
                # st.write(f"Image: {image_path}")
                st.image(display_ground_truth_image(image_path), caption=f"Image: {relative_path} ")
                
    with st.expander("PERFORMANCE METRICS"):
        st.subheader('Latency Graph')
        df = pd.DataFrame(model_metrics['latency']) * 100
        fig = px.line(df, labels={'value': 'Latency (in ms)', 'index': 'Images'})
        fig.update_xaxes(visible=False)  
        st.plotly_chart(fig, use_container_width=True)

        st.subheader('Precision Graph')
        df_precision = pd.DataFrame(model_metrics['precision']) 
        fig_precision = px.line(df_precision, labels={'value': 'Precision %', 'index': 'Images'})
        fig_precision.update_xaxes(visible=False)
        st.plotly_chart(fig_precision, use_container_width=True)

        st.subheader('Recall Graph')
        df_recall = pd.DataFrame(model_metrics['recall']) 
        fig_recall = px.line(df_recall, labels={'value': 'Recall %', 'index': 'Images'})
        fig_recall.update_xaxes(visible=False)
        st.plotly_chart(fig_recall, use_container_width=True)

        st.subheader('F1 Score Graph')
        df_f1 = pd.DataFrame(model_metrics['f1_score']) 
        fig_f1 = px.line(df_f1, labels={'value': 'F1 Score %', 'index': 'Images'})
        fig_f1.update_xaxes(visible=False)
        st.plotly_chart(fig_f1, use_container_width=True)

        st.subheader('True Positive Graph')
        df_f1 = pd.DataFrame(model_metrics['true_positive']) 
        fig_f1 = px.bar(df_f1, labels={'value': 'True Positive', 'index': 'Images'})
        fig_f1.update_xaxes(visible=False)
        st.plotly_chart(fig_f1, use_container_width=True)

        st.subheader('False Positive Graph')
        df_f1 = pd.DataFrame(model_metrics['false_positive']) 
        fig_f1 = px.bar(df_f1, labels={'value': 'False Positive', 'index': 'Images'})
        fig_f1.update_xaxes(visible=False)
        st.plotly_chart(fig_f1, use_container_width=True)

        st.subheader('False Negative Graph')
        df_f1 = pd.DataFrame(model_metrics['false_negative']) 
        fig_f1 = px.bar(df_f1, labels={'value': 'False Negative', 'index': 'Images'})
        fig_f1.update_xaxes(visible=False)
        st.plotly_chart(fig_f1, use_container_width=True)
        
        

        # st.subheader('Confusion Matrix')
        # for model in selected_models:
        #     st.write(f"{model} Confusion Matrix")
        #     cm = model_metrics['confusion_matrix'][model]
        #     labels = coco_labels if "YoloV5" not in model else yolov5_labels
        #     label_names = [labels[i] for i in range(len(labels))]
        #     fig_cm = ff.create_annotated_heatmap(z=cm, x=label_names, y=label_names, colorscale='Viridis')
        #     st.plotly_chart(fig_cm, use_container_width=True)

# Main function
def main():
    st.set_page_config(page_title="AI CV V&V Performance Toolset", layout="wide")
    st.title("Artificial Intelligence V&V Performance Toolset (Computer Vision)")
    system_info()

    models = ["NN 1 (YoloV5)", "NN 2 (Faster RCNN)", "NN 3 (Mask RCNN)", "NN 4 (SSD)", "NN 5 (RetinaNet)"]
    cols = st.columns(len(models))
    selected_models = []
    for idx, model in enumerate(models):
        if cols[idx].checkbox(model):
            selected_models.append(model)

    dataset_path = st.text_input("Dataset", value='datasets/civilian_data')
    if st.button("Calculate performance metrics ▶️"):
        calc_metrics(selected_models, dataset_path)

if __name__ == "__main__":
    main()
