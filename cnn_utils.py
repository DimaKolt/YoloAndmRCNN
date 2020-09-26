import os
import subprocess
from our_paths import *

class class_names:
    def __init__(self):
        self.names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
                               'bus', 'train', 'truck', 'boat', 'traffic light',
                               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
                               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
                               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
                               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
                               'kite', 'baseball bat', 'baseball glove', 'skateboard',
                               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
                               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
                               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
                               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
                               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
                               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
                               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
                               'teddy bear', 'hair drier', 'toothbrush']

    def str2int(self, str):
        for i, name in enumerate(self.names):
            name = conv2one_name(name)
            if (name == str):
                return i

    def int2str(self, int):
        return conv2one_name(self.names[int])



def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = abs(max((xB - xA, 0)) * max((yB - yA), 0))
    if interArea == 0:
        return 0
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = abs((boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    boxBArea = abs((boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou


def convert_to_yolo_pred(mrcnn_pred):
    rois = mrcnn_pred['rois']
    class_ids = mrcnn_pred['class_ids']
    scores = mrcnn_pred['scores']
    class_names = mrcnn_pred['class_names']
    ress = []
    for i in range(len(rois)):
        ress.append(  # TODO class ids -1
            [class_names[i], scores[i], round(rois[i][1]), round(rois[i][0]), round(rois[i][3]),
             round(rois[i][2]), "MRCNN"])
    return ress


class ResultFilter:
    def __init__(self, yolo_pred, mrcnn_pred):
        self.yolo_pred = yolo_pred.copy()
        self.mrcnn_pred = convert_to_yolo_pred(mrcnn_pred)

    def filter_IOU_and_score(self, iou_bar):
        yolo_pred_temp = self.yolo_pred.copy()
        mr_pred_temp = self.mrcnn_pred.copy()

        for obj_yolo in self.yolo_pred:
            for obj_mr in self.mrcnn_pred:
                box_yolo = [obj_yolo[2], obj_yolo[3], obj_yolo[4], obj_yolo[5]]
                box_mr = [obj_mr[2], obj_mr[3], obj_mr[4], obj_mr[5]]
                iou = bb_intersection_over_union(box_yolo, box_mr)
                if iou >= iou_bar:
                    if obj_yolo[1] >= obj_mr[1] and obj_mr in mr_pred_temp:
                        mr_pred_temp.remove(obj_mr)
                    elif obj_yolo in yolo_pred_temp:
                        yolo_pred_temp.remove(obj_yolo)
                    break
        return yolo_pred_temp + mr_pred_temp

    def filter_IOU_and_score_and_conf(self, iou_bar, conf_bar):
        yolo_pred_temp = self.yolo_pred.copy()
        mr_pred_temp = self.mrcnn_pred.copy()

        for obj_yolo in self.yolo_pred:
            for obj_mr in self.mrcnn_pred:
                box_yolo = [obj_yolo[2], obj_yolo[3], obj_yolo[4], obj_yolo[5]]
                box_mr = [obj_mr[2], obj_mr[3], obj_mr[4], obj_mr[5]]
                iou = bb_intersection_over_union(box_yolo, box_mr)
                if iou >= iou_bar:
                    if obj_yolo[1] >= obj_mr[1]:
                        mr_pred_temp.remove(obj_mr)
                    else:
                        yolo_pred_temp.remove(obj_yolo)
                    break
        merged_arr = yolo_pred_temp + mr_pred_temp
        merged_arr_cpy = merged_arr.copy()
        for obj in merged_arr:
            if obj[1] < conf_bar:
                merged_arr_cpy.remove(obj)
        return merged_arr_cpy


def div2sets(yolo_result_path, mrcnn_result_path):
    with open(yolo_result_path, 'r') as f:
        yolo_pred = f.readlines()

    with open(mrcnn_result_path, 'r') as f:
        mrcnn_pred = f.readlines()

    yolo_pred_temp = yolo_pred.copy()
    mr_pred_temp = mrcnn_pred.copy()

    pairs_list = []
    pairs_box_list = []
    for obj_yolo in yolo_pred:
        for obj_mr in mrcnn_pred:
            obj_yolo_splited = obj_yolo.split()
            obj_mr_splited = obj_mr.split()
            box_yolo = [int(obj_yolo_splited[2]), int(obj_yolo_splited[3]), int(obj_yolo_splited[4]),
                        int(obj_yolo_splited[5])]
            box_mr = [int(obj_mr_splited[2]), int(obj_mr_splited[3]), int(obj_mr_splited[4]), int(obj_mr_splited[5])]
            iou = bb_intersection_over_union(box_yolo, box_mr)
            if iou >= 0.5:
                pairs_list.append([pair_obj_conv(obj_yolo_splited), pair_obj_conv(obj_mr_splited)])
                pairs_box_list.append([[obj_yolo_splited[0]] + box_yolo, [obj_mr_splited[0]] + box_mr])
                if obj_mr in mr_pred_temp:
                    mr_pred_temp.remove(obj_mr)
                if obj_yolo in yolo_pred_temp:
                    yolo_pred_temp.remove(obj_yolo)

    solo_list = yolo_pred_temp + mr_pred_temp
    solo_box_list = []
    for solo_obj in solo_list:
        solo_obj = solo_obj.split()
        solo_box_list.append([solo_obj[0], int(solo_obj[2]), int(solo_obj[3]), int(solo_obj[4]),
                              int(solo_obj[5])])
    solo_list_converted = list(map(solo_obj_conv, solo_list))
    return pairs_list, pairs_box_list, solo_list_converted, solo_box_list


def pair_obj_conv(obj):
    return [class_names().str2int(obj[0]), float(obj[1]), int(obj[4]) - int(obj[2]), int(obj[5]) - int(obj[3])]


def solo_obj_conv(obj):
    obj = obj.split()
    return [class_names().str2int(obj[0]), float(obj[1]), int(obj[4]) - int(obj[2]), int(obj[5]) - int(obj[3]),
            algorithemStringToNumber(obj[6])]


def algorithemStringToNumber(name):
    if name == 'MRCNN':
        return 1
    else:
        return 0


def chooseAlgoUsingIOU(yolo_box, mrcnn_box, ground_truth):
    max_yolo_alg_0 = 0
    max_mrcnn_alg_1 = 0
    for pred in ground_truth:
        pred_box = [int(pred[1]), int(pred[2]), int(pred[3]),
                        int(pred[4])]
        temp_alg0 = bb_intersection_over_union(yolo_box[1:], pred_box)
        temp_alg1 = bb_intersection_over_union(mrcnn_box[1:], pred_box)
        if (temp_alg0 > max_yolo_alg_0) and class_names().str2int(yolo_box[0]) == class_names().str2int(pred[0]):
            max_yolo_alg_0 = temp_alg0
        if (temp_alg1 > max_mrcnn_alg_1) and class_names().str2int(mrcnn_box[0]) == class_names().str2int(pred[0]):
            max_mrcnn_alg_1 = temp_alg1

    if (max_yolo_alg_0 >= max_mrcnn_alg_1):
        return 0
    else:
        return 1

def checkIfPredictionIsRight(pred_box, ground_truth):
    for real_pred in ground_truth:
        if(class_names().str2int(pred_box[0]) == class_names().str2int(real_pred[0])) :
            iou = bb_intersection_over_union(pred_box[1:], real_pred[1:])
            if(iou>= 0.5) :
                return 1
    return 0

def savePredictionsToFile(path, file_name, predictions):
    if not os.path.exists(path):
        os.makedirs(path)
    with open(os.path.join(path, file_name), 'w+') as filehandle:
        for line in predictions:
            x = line[0]
            if len((x.split())) > 1:
                line[0] = line[0].replace(" ", "_")
                line[0] = line[0].lower()
            filehandle.write(' '.join([str(x) for x in line]))
            filehandle.write('\n')

def conv2one_name(name):
    if len((name.split())) > 1:
        name = name.replace(" ", "_")
        name = name.lower()
    return name

def calculate_map(pred_path):
    cmd = 'python ' + calc_map_script_path + ' -np -q --ground_true_path ' + ground_truth_test_path + ' --predict_path ' + pred_path
    # TODO convert map to class or copy to our env.
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, shell=True)
    # koko = proc.communicate()[0]
    with open(map_output_path, 'a+') as filehandle:
        filehandle.write("map of: " + pred_path + " is:\n" + str(proc.communicate()[0]) + " \n\n")
