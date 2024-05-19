import numpy as np
from sklearn.metrics import confusion_matrix
import torch

def calculate_iou(box1, box2):
    # Calculate the intersection
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # Calculate the areas of the boxes
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    # Calculate the union
    union_area = box1_area + box2_area - intersection_area

    return intersection_area / union_area

def calculate_metrics(ground_truth, predictions, iou_threshold=0.5):
    gt_boxes = np.array(ground_truth['boxes'])
    gt_labels = np.array(ground_truth['labels'])
    
    pred_boxes = predictions['boxes'].numpy()
    pred_labels = predictions['labels'].numpy()

    # Match predictions to ground truth
    tp, fp, fn = 0, 0, 0
    matched_gt = set()
    matched_pred = set()
    
    for i, gt_box in enumerate(gt_boxes):
        for j, pred_box in enumerate(pred_boxes):
            iou = calculate_iou(gt_box, pred_box)
            if iou >= iou_threshold and gt_labels[i] == pred_labels[j]:
                tp += 1
                matched_gt.add(i)
                matched_pred.add(j)
                break
    
    fp = len(pred_boxes) - len(matched_pred)
    fn = len(gt_boxes) - len(matched_gt)

    # Calculate metrics
    accuracy = tp / (tp + fp + fn) if (tp + fp + fn) != 0 else 0

    precision = tp / (tp + fp) if (tp + fp) != 0 else 0
    recall = tp / (tp + fn) if (tp + fn) != 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0

    # Calculate confusion matrix
    y_true = []
    y_pred = []
    for i, gt_label in enumerate(gt_labels):
        if i in matched_gt:
            y_true.append(gt_label)
            y_pred.append(gt_label)
        else:
            y_true.append(gt_label)
            y_pred.append(0)  # No prediction for this ground truth

    for j, pred_label in enumerate(pred_labels):
        if j not in matched_pred:
            y_true.append(0)  # No ground truth for this prediction
            y_pred.append(pred_label)

    conf_matrix = confusion_matrix(y_true, y_pred)

    # Calculate IoU
    ious = []
    for i, gt_box in enumerate(gt_boxes):
        for j, pred_box in enumerate(pred_boxes):
            iou = calculate_iou(gt_box, pred_box)
            ious.append(iou)
    mean_iou = np.mean(ious) if ious else 0.0

    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'mean_iou': mean_iou,
        'confusion_matrix': conf_matrix.tolist()
    }
    
    return metrics


# Example usage
ground_truth = {
    'boxes': [
        [322.91, 521.22, 396.09, 569.21], [387.2, 522.46, 443.89, 561.22], [243.11, 516.36, 276.02, 542.46],
        [456.64, 497.91, 638.43, 628.17], [220.57, 514.67, 240.4, 529.87], [118.53, 514.51, 130.19, 543.26],
        [144.97, 513.06, 152.87, 539.55], [152.94, 516.41, 160.22, 539.2], [183.69, 515.55, 191.53, 533.71],
        [166.49, 516.94, 174.76, 542.43], [388.44, 517.91, 394.4, 527.27], [209.23, 514.3, 228.11, 529.05],
        [306.15, 485.66, 362.91, 523.3], [190.69, 495.94, 209.17, 528.01], [157.91, 518.04, 163.37, 541.55],
        [460.78, 497.1, 639.23, 632.5]
    ],
    'labels': [3, 3, 3, 3, 3, 1, 1, 1, 1, 1, 1, 3, 6, 8, 1, 8]
}

predictions = {
    'boxes': torch.tensor([
        [259.22940, 418.41061, 315.79767, 457.70648], [310.57520, 419.15338, 352.70557, 450.64105],
        [95.37569, 414.27142, 104.42223, 437.35309], [192.62138, 416.11783, 222.31432, 436.22980],
        [176.08458, 413.53885, 192.86310, 425.82175], [376.29388, 400.00726, 512.00000, 504.59476],
        [124.71800, 414.57269, 130.65388, 434.45010], [168.54501, 340.87042, 188.82661, 358.22330],
        [115.92359, 415.02908, 121.49275, 433.97574], [175.45097, 413.78357, 185.12437, 424.41248],
        [127.48483, 416.12827, 132.94164, 434.72757], [122.95686, 415.76089, 128.35451, 434.02612],
        [129.74051, 420.38635, 135.49808, 434.97559], [264.42935, 388.53461, 292.85764, 428.35718],
        [190.81651, 415.07324, 205.00667, 427.97064], [154.42665, 407.25089, 167.52829, 423.59058],
        [146.48676, 414.94009, 155.04054, 424.11542], [119.11640, 416.48785, 124.56265, 433.72400],
        [367.98123, 400.39279, 511.17047, 503.65662], [153.72717, 406.54388, 167.69614, 423.68695],
        [268.54071, 390.46646, 291.00235, 410.63828], [106.86730, 417.23499, 111.95693, 434.09695],
        [310.43774, 414.99991, 317.63812, 427.36810], [185.02681, 414.19656, 195.99110, 426.33264],
        [113.75820, 417.13959, 119.00644, 434.03961], [166.58344, 304.35464, 191.51855, 350.44098],
        [188.76425, 414.99667, 197.80449, 427.76276], [104.86803, 417.99985, 110.17202, 434.28757],
        [93.91432, 415.24307, 101.25194, 434.94128], [133.71587, 417.72900, 138.38356, 435.41238],
        [216.43039, 418.38806, 223.13159, 426.23621], [263.41650, 387.35043, 267.64597, 395.06433],
        [203.21980, 409.36960, 214.69171, 415.15109], [146.71939, 415.09995, 151.09857, 425.39706],
        [217.71205, 412.79700, 223.56837, 421.84769], [131.15562, 423.61514, 136.65518, 435.30093],
        [166.55635, 344.61108, 176.12236, 359.38803], [166.35490, 304.50742, 190.87865, 341.91754],
        [263.89346, 388.03113, 269.17502, 397.52713], [101.61252, 441.67252, 107.07720, 447.44559],
        [486.50812, 419.13187, 512.00000, 486.04172], [262.48239, 388.15591, 266.91592, 398.45905],
        [215.05354, 413.69510, 224.15828, 424.15854], [220.27013, 420.24622, 225.67474, 428.93588],
        [193.58226, 415.69662, 214.52379, 428.28546], [166.52658, 413.98474, 174.53056, 423.87982],
        [167.36279, 414.33334, 181.29013, 424.13177], [217.85318, 404.36182, 223.80798, 410.57175],
        [135.48460, 419.54245, 139.14790, 435.58722], [187.10097, 411.36417, 203.84480, 424.07855],
        [162.41946, 399.11917, 166.56197, 406.58112]
    ]),
    'labels': torch.tensor([3, 3, 1, 3, 3, 3, 1, 10, 1, 3, 1, 1, 1, 6, 3, 3, 3, 1, 8, 8, 6, 1, 1, 3, 1, 10, 3, 1, 1, 1, 3, 10, 3, 1, 3, 1, 10, 13, 10, 41, 3, 10, 3, 3, 3, 3, 3, 3, 1, 3, 10])
}
metrics = calculate_metrics(ground_truth, predictions)
print(metrics)
