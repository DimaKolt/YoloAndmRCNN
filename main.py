from combCNN import CombinedNet as my_net
from MAPCalculator import MAPCalculator
import cnn_utils

import os
pics_path = "/home/dima/Downloads/train2014/"
ground_truth_path = "/home/dima/YoloAndmRCNN/groundtruth/"
prediction_path = "/home/dima/YoloAndmRCNN/prediction/iou70/"
# chosen_pic = "COCO_train2014_000000581921.jpg"


calc = MAPCalculator()
combinedNet = my_net()
for i, chosen_pic in enumerate(os.listdir(pics_path)):
    if i == 1000:
        break

    real_box = calc.getGroundTruth(chosen_pic)
    cnn_utils.savePredictionsToFile(ground_truth_path + str(calc.imageNameToId(chosen_pic)) + '.txt', real_box)


    combinedNet.readImage(pics_path + chosen_pic)
    combinedNet.predict()
    res = combinedNet.filterByIOU(0.7)

    cnn_utils.savePredictionsToFile(prediction_path + str(calc.imageNameToId(chosen_pic)) + '.txt', res)
    #
    # combinedNet.show(res)
    print("12334")
    print("finished pic %s" % chosen_pic)

coco = 7
