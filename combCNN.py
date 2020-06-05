import Mask_RCNN_env.mRCNN.MaskRCNNNet as my_rcnn
import YOLO_env.YOLONet as my_yolo

mrcnn_try1 = my_rcnn.My_mRCNN()
mrcnn_try1.readImage("/home/dima/YoloAndmRCNN/images/9247489789_132c0d534a_z.jpg")
ress = mrcnn_try1.predict()
mrcnn_try1.show(ress) #TODO run with pic

yolo_try1 = my_yolo.YOLONet()
yolo_try1.readImage("/home/dima/YoloAndmRCNN/images/9247489789_132c0d534a_z.jpg")
yolo_try1.predict()
yolo_try1.show()

coco = 7
