import pickle
from sklearn import tree
import graphviz

import cnn_utils
from our_paths import *
import os
from cnn_utils import div2sets, savePredictionsToFile, checkIfPredictionIsRight

# filename = 'tree_pairs_pkl'

with open(filename_pairs, 'rb') as file:
    tree_pairs = pickle.load(file)

# dot_data = tree.export_graphviz(tree_pairs, out_file=None)
# graph = graphviz.Source(dot_data)
# graph.render("our_tree")

with open(filename_solo, 'rb') as file:
    tree_solo = pickle.load(file)

with open(filename_solo_net, 'rb') as file:
    net_solo = pickle.load(file)

for i, chosen_pic in enumerate(os.listdir(yolo_test_set_path)):
    pair_list, pair_boxes, solo_list, solo_boxes = div2sets(os.path.join(yolo_test_set_path, chosen_pic),
                                                            os.path.join(mrcnn_test_set_path, chosen_pic))
    list_of_pair_objects = []
    list_of_solo_objects = []

    for i, pair_box in enumerate(pair_boxes):
        list_of_pair_objects.append(pair_list[i][0] + pair_list[i][1])

    for i, solo_box in enumerate(solo_list):
        list_of_solo_objects.append(solo_list[i])

    obj_list = []
    obj_list_net = []
    if len(list_of_pair_objects) is not 0:
        pair_result_per_pic = tree_pairs.predict(list_of_pair_objects)
        for obj_i, res in enumerate(pair_result_per_pic):
            obj_list.append(
                [pair_boxes[obj_i][res][0], pair_list[obj_i][res][1], pair_boxes[obj_i][res][1],
                 pair_boxes[obj_i][res][2],
                 pair_boxes[obj_i][res][3], pair_boxes[obj_i][res][4]])

    obj_list_net = obj_list.copy()

    if len(list_of_solo_objects) is not 0:
        solo_result_per_pic = tree_solo.predict(list_of_solo_objects)
        for obj_i, res in enumerate(solo_result_per_pic):
            if res:
                obj_list.append([solo_boxes[obj_i][0], solo_list[obj_i][1], solo_boxes[obj_i][1], solo_boxes[obj_i][2],
                                 solo_boxes[obj_i][3], solo_boxes[obj_i][4]])
        solo_result_per_pic = net_solo.predict(list_of_solo_objects)
        for obj_i, res in enumerate(solo_result_per_pic):
            if res:
                obj_list_net.append(
                    [solo_boxes[obj_i][0], solo_list[obj_i][1], solo_boxes[obj_i][1], solo_boxes[obj_i][2],
                     solo_boxes[obj_i][3], solo_boxes[obj_i][4]])

    cnn_utils.savePredictionsToFile(result_decision_trees_path, chosen_pic, obj_list)

    cnn_utils.savePredictionsToFile(result_decision_trees_and_net_path, chosen_pic, obj_list_net)

    coco = 9

coco = 0
