from combCNN import CombinedNet as my_net
from MAPCalculator import MAPCalculator
import cnn_utils

pics_path = "/home/dima/Downloads/train2014/"
ground_truth_path = "/home/dima/YoloAndmRCNN/groundtruth/"
prediction_path = "/home/dima/YoloAndmRCNN/prediction/"
chosen_pic = "COCO_train2014_000000581921.jpg"


calc = MAPCalculator()
real_box = calc.getGroundTruth("COCO_train2014_000000581921.jpg")
cnn_utils.savePredictionsToFile(ground_truth_path + str(calc.imageNameToId(chosen_pic)) + '.txt', real_box)


combinedNet = my_net()
combinedNet.readImage(pics_path + chosen_pic)
combinedNet.predict()
res = combinedNet.filterByIOU(0.7)

cnn_utils.savePredictionsToFile(prediction_path + str(calc.imageNameToId(chosen_pic)) + '.txt', res)
#
combinedNet.show(res)

coco = 7
