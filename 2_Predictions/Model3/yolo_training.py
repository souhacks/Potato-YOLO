# -*- coding: utf-8 -*-
"""yolo_training.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1nYlZnC_VT_seARQeyBclpIgEdq0fAhFc
"""

import os
os.chdir('/content/drive/MyDrive/Yolo_training')

ls

!git clone https://github.com/ultralytics/yolov5.git

os.chdir('yolov5')

ls

!pip install -r requirements.txt

"""**Training YOLO v5 model**"""

!python train.py --data data.yaml --cfg yolov5s.yaml --batch-size 20 --name Model --epoch 80

!python export.py --weights runs/train/Model3/weights/best.pt --include torchscript onnx