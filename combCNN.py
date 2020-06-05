import Mask_RCNN.mRCNN.test_mask_rcnn as my_rcnn

mrcnn_try1 = my_rcnn.My_mRCNN()
mrcnn_try1.readImage("/home/dima/YoloAndmRCNN/Mask_RCNN/images/9247489789_132c0d534a_z.jpg")
ress = mrcnn_try1.predict()
mrcnn_try1.show(ress)

coco = 7
