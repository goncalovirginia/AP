from ultralytics import YOLO
from os import listdir
import os.path

# Object detection model

yolo = YOLO('yolo.pt')
yolo.to('cuda')

# Predict bbox and save cropped images

def predict_bbox_save_crop(input_dir, output_dir):
    for image in listdir(input_dir):
        results = yolo.predict(source=input_dir+image)[0]
        results.save_crop(save_dir=output_dir, file_name=os.path.splitext(image)[0])

# Run stuff

predict_bbox_save_crop('images/train/', 'images/train/')
predict_bbox_save_crop('images/test/', 'images/test/')
