from combCNN import CombinedNet as my_net
from MAPCalculator import MAPCalculator
import cnn_utils

import os
pics_path = "/home/dima/Downloads/train2014/"
ground_truth_path = "/home/dima/YoloAndmRCNN/groundtruth/"
prediction_path = "/home/dima/YoloAndmRCNN/prediction/"
# folder = "filterByHighestScoreTakeAll_iou70/"
# chosen_pic = "COCO_train2014_000000581921.jpg"


calc = MAPCalculator()
combinedNet = my_net()
for i, chosen_pic in enumerate(os.listdir(pics_path)):
    if i == 354:
        break
    print("start pic %s" % chosen_pic)

    real_box = calc.getGroundTruth(chosen_pic)
    cnn_utils.savePredictionsToFile(ground_truth_path, str(calc.imageNameToId(chosen_pic)) + '.txt', real_box)


    combinedNet.readImage(pics_path + chosen_pic)
    combinedNet.predict()
    result_dict = {}
    result_dict["res_ByHighestScoreTakeAllIou80"] = combinedNet.filterByHighestScoreTakeAll(0.8)
    result_dict["res_ByHighestScoreTakeAllIou70"] = combinedNet.filterByHighestScoreTakeAll(0.7)
    result_dict["res_ByHighestScoreTakeAllIou60"] = combinedNet.filterByHighestScoreTakeAll(0.6)
    result_dict["res_ByHighestScoreTakeAllIou50"] = combinedNet.filterByHighestScoreTakeAll(0.5)
    result_dict["res_ByHighestScoreTakeWithConf90"] = combinedNet.filterByHighestScoreTakeWithConf(0.7, 0.90)
    result_dict["res_ByHighestScoreTakeWithConf80"] = combinedNet.filterByHighestScoreTakeWithConf(0.7, 0.80)
    result_dict["res_ByHighestScoreTakeWithConf70"] = combinedNet.filterByHighestScoreTakeWithConf(0.7, 0.70)
    result_dict["res_yolo_only"] = combinedNet.getYoloRes()
    result_dict["res_mrnn_only"] = combinedNet.getMrcnnRess()

    for kye, ress in result_dict.items():
        cnn_utils.savePredictionsToFile(prediction_path + str(kye), str(calc.imageNameToId(chosen_pic)) + '.txt',
                                        ress)
    # combinedNet.show(res)
    print("finished pic %s" % chosen_pic)

coco = 7
