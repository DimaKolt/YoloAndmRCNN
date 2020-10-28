import subprocess

from combCNN import CombinedNet as my_net
from MAPCalculator import MAPCalculator
import cnn_utils
from cnn_utils import div2sets

import os
from our_paths import *

calc = MAPCalculator()
combinedNet = my_net()
result_dict = {}
for i, chosen_pic in enumerate(os.listdir(pics_path)):
    #choose num of pic to work on
    if i<7000:
        continue
    if i==7120: #7000
        break

    print("start pic %s " % chosen_pic + " i= " + str(i))

    if os.path.exists(os.path.join(ground_truth_test_path, str(calc.imageNameToId(chosen_pic)) + '.txt')):
        continue

    real_box = calc.getGroundTruth(chosen_pic)
    cnn_utils.savePredictionsToFile(ground_truth_test_path, str(calc.imageNameToId(chosen_pic)) + '.txt', real_box)

    combinedNet.readImage('/home/dima/YoloAndmRCNN/images/' + 'snowboard.jpg')
    try:
        combinedNet.predict()
    except:
        print("exception %s", i)
        os.remove(os.path.join(ground_truth_test_path, str(calc.imageNameToId(chosen_pic)) + '.txt'))
        continue
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
        cnn_utils.savePredictionsToFile(prediction_test_path + str(kye), str(calc.imageNameToId(chosen_pic)) + '.txt',
                                        ress)

        combinedNet.show(ress)
    print("finished pic %s" % chosen_pic)

coco = 7


