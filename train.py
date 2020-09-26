from cnn_utils import div2sets, chooseAlgoUsingIOU, checkIfPredictionIsRight
import os
from MAPCalculator import MAPCalculator

from sklearn import tree, preprocessing
from sklearn.neural_network import MLPClassifier

import graphviz

import pickle
from our_paths import *


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
        x_pairs.append(pair_list[i][0] + pair_list[i][1])
        y_pairs.append(alg_index)

    for i, solo_box in enumerate(solo_boxes):
        is_exist = checkIfPredictionIsRight(solo_box, real_boxes)
        x_solo.append(solo_list[i])
        y_solo.append(is_exist)

clf_pair = tree.DecisionTreeClassifier()
clf_pair = clf_pair.fit(x_pairs, y_pairs)

clf_solo = tree.DecisionTreeClassifier()
clf_solo = clf_solo.fit(x_solo, y_solo)

clf_net = MLPClassifier(solver='sgd', alpha=1e-5,
                    hidden_layer_sizes=(100,), random_state=1, max_iter=4000)

X_scaled = preprocessing.scale(x_solo)

clf_net.fit(X_scaled, y_solo)

with open(filename_solo_net, 'wb') as file:
    pickle.dump(clf_net, file)

with open(filename_pairs, 'wb') as file:
    pickle.dump(clf_pair, file)


with open(filename_solo, 'wb') as file:
    pickle.dump(clf_solo, file)

# dot_data_pairs = tree.export_graphviz(clf_pair, out_file=None)
# graph_pairs = graphviz.Source(dot_data_pairs)
# graph_pairs.render(graph_pairs_name, graph_path)
#
# dot_data_solo = tree.export_graphviz(clf_solo, out_file=None)
# graph_solo = graphviz.Source(dot_data_solo)
# graph_solo.render(graph_solo_name, graph_path)

coco = 0
