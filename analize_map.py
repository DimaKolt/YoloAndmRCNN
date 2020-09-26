import os

from cnn_utils import calculate_map
from our_paths import *

for pred in os.listdir(prediction_test_path):
    coco=0

    calculate_map(prediction_test_path + pred)

