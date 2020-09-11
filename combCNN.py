import Mask_RCNN_env.mRCNN.MaskRCNNNet as mrcnn
import YOLO_env.YOLONet as yolo
import cnn_utils as utils

class CombinedNet:

    def __init__(self):
        self.YOLO_net = yolo.YOLONet()
        self.MRCNN_net = mrcnn.My_mRCNN()
        self.YOLO_res = []
        self.MRCNN_res = []
        self.image_path = ""

    def readImage(self,path):
        self.MRCNN_net.readImage(path)
        self.YOLO_net.readImage(path)

    def predict(self):
        self.YOLO_res = self.YOLO_net.predict()
        self.MRCNN_res = self.MRCNN_net.predict()

    def getYoloRes(self):
        return self.YOLO_res

    def getMrcnnRess(self):
        return utils.convert_to_yolo_pred(self.MRCNN_res)

    def filterByHighestScoreTakeAll(self, IOU_bar):
        my_predicts = utils.ResultFilter(self.YOLO_res, self.MRCNN_res)
        return my_predicts.filter_IOU_and_score(IOU_bar)

    def filterByHighestScoreTakeWithConf(self, IOU_bar, confd_bar):
        my_predicts = utils.ResultFilter(self.YOLO_res, self.MRCNN_res)
        return my_predicts.filter_IOU_and_score_and_conf(IOU_bar, confd_bar)

    # def div2pairs(self):
    #     my_predicts = utils.ResultFilter(self.YOLO_res, self.MRCNN_res)
    #     return my_predicts.div2sets()

    def show(self, predictions):
        self.YOLO_net.show(predictions)
        # mrcnn_try1.show(mrcnn_ress) #TODO non blocking pic


