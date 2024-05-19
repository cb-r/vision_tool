import torch
import os

# Model loading
model = torch.hub.load(r'/Users/siravindran/programming/cv_performance/yolov5', 'custom', path=r'yolov5s.pt', source='local')

# Inference on images
img = "https://ultralytics.com/images/zidane.jpg"  # Can be a file, Path, PIL, OpenCV, numpy, or list of images

# Run inference
results = model(img)

# Display results
results.show()  # Other options: .show(), .save(), .crop(), .pandas(), etc.