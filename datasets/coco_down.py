import os
import requests
from pycocotools.coco import COCO
import json

# Configuration
data_dir = 'civilian_data'
images_dir = os.path.join(data_dir, 'images')
annotations_dir = os.path.join(data_dir, 'annotations')
num_images = 20

# Create directories if they don't exist
os.makedirs(images_dir, exist_ok=True)
os.makedirs(annotations_dir, exist_ok=True)

# Initialize COCO API for instance annotations
coco = COCO('instances_val2017.json')

# Get category IDs for car, bike, and person
catIds = coco.getCatIds(catNms=['car', 'truck', 'person', 'aircraft', 'ship', 'bus'])
print(f"Category IDs for car, bicycle, person: {catIds}")

# Get all image IDs containing the above categories
imgIds = coco.getImgIds(catIds=catIds)
print(f"Found {len(imgIds)} images with the specified categories.")

# Limit to the first num_images images
imgIds = imgIds[:num_images]
images = coco.loadImgs(imgIds)

print(f"Downloading {len(images)} images...")

for img in images:
    try:
        # Download image
        print(f"Downloading image {img['file_name']} from {img['coco_url']}")
        img_data = requests.get(img['coco_url']).content
        with open(os.path.join(images_dir, img['file_name']), 'wb') as handler:
            handler.write(img_data)
        
        # Get annotations
        annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
        anns = coco.loadAnns(annIds)
        
        # Save annotations
        annotation_file = os.path.join(annotations_dir, f"{img['file_name'].split('.')[0]}.json")
        print(f"Saving annotations to {annotation_file}")
        with open(annotation_file, 'w') as f:
            json.dump(anns, f)
    except Exception as e:
        print(f"Error downloading or saving {img['file_name']}: {e}")

print("Download complete!")
