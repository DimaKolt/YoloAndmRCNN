from cnn_utils import div2sets, chooseAlgoUsingIOU, checkIfPredictionIsRight
import os
from MAPCalculator import MAPCalculator

yolo_results_path = "/home/dima/YoloAndmRCNN/prediction/res_yolo_only"
mrcnn_results_path = "/home/dima/YoloAndmRCNN/prediction/res_mrnn_only"
ground_truth_path = "/home/dima/YoloAndmRCNN/groundtruth/"

calc = MAPCalculator()
x_pairs = []
y_pairs = []

x_solo = []
y_solo = []

for i, chosen_pic in enumerate(os.listdir(yolo_results_path)):
    pair_list, pair_boxes, solo_list, solo_boxes = div2sets(os.path.join(yolo_results_path, chosen_pic),
                                    os.path.join(mrcnn_results_path, chosen_pic))

    real_boxes = calc.getGroundTruth(chosen_pic)
    for i, pair_box in enumerate(pair_boxes):
        alg_index = chooseAlgoUsingIOU(pair_box[0], pair_box[1], real_boxes)
        x_pairs.append(pair_list[i])
        y_pairs.append(alg_index)

    for i, solo_box in enumerate(solo_boxes):
        is_exist = checkIfPredictionIsRight(solo_box, real_boxes)
        x_solo.append(solo_list[i])
        y_solo.append(is_exist)

coco = 0

