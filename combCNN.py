import Mask_RCNN_env.mRCNN.MaskRCNNNet as my_rcnn
import YOLO_env.YOLONet as my_yolo
import cnn_utils as utils


choosen_pic = "/home/dima/YoloAndmRCNN/images/4410436637_7b0ca36ee7_z.jpg"

mrcnn_try1 = my_rcnn.My_mRCNN()
mrcnn_try1.readImage(choosen_pic)
mrcnn_ress = mrcnn_try1.predict()


yolo_try1 = my_yolo.YOLONet()
yolo_try1.readImage(choosen_pic)
yolo_ress = yolo_try1.predict()

my_predicts = utils.ResultFilter(yolo_ress, mrcnn_ress)
filtred_by_iou = my_predicts.filter_IOU_and_score(0.80)

# yolo_try1.show(yolo_ress)
yolo_try1.show(filtred_by_iou)
# mrcnn_try1.show(mrcnn_ress) #TODO non blocking pic

coco = 7
