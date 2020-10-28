# Imporved object detection Project using YOLO and MRCNN algorithems 

Project description:
Object detection is a Computer Vision task which requires to find the locations of all objects in a given image and classify the objects. For example: https://towardsdatascience.com/object-detection-with-10-lines-of-code-d6cb4d86f606


In this project we examine several approaches to DNNs (YOLO, MRCNN, RESNET50) for object detection and then build a system that combines several approaches to one large and highly-accurate system.


Setup:
- get yolov3.weights in YOLO_env dir.
- get mask_rcnn_coco.h5 (or any other wights for mrcnn) in Mask_RCNN_env dir.
- download pics and annonetions and update path to that folder in pics_path.py.

How to run:
  - create the databse and use naive decision rules, for that run create_dataset_and_naived.py.
  - train the decision tree and DNN, run train.py.
  - run predict.py.
  - measure the results, run calc_map.py.
