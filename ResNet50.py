import keras
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing import image
from keras.applications.resnet50 import ResNet50

import numpy as np
import tensorflow as tf

graph = tf.get_default_graph()

import cv2

class MyResNet50(object):
    def __init__(self):
        # self.model = keras.Sequential()
        # # self.model = ResNet50()
        # self.model.add(keras.layers.ZeroPadding2D((90, 90), input_shape=(292-245, 307-255, 3)))
        # self.model.add(ResNet50())
        self.model = ResNet50(weights='imagenet')


    def predict(self, box, img_path):
        img = cv2.imread(img_path)
        # cv2.imwrite("temp.jpg", img)
        # cv2.imshow("not cropped", img)
        crop_img = img[self.round2zero(box[2]):self.round2zero(box[4]), self.round2zero(box[1]):self.round2zero(box[3])]
        l_r_padding = round((197 - (box[4] - box[2])))
        u_d_padding = round((197 - (box[3] - box[1])))

        # if l_r_padding > 80 and u_d_padding > 80:
        #     return
        # if box[4] - box[2] < 255 or box[3] - box[1] < 255:
        #     return


        # self.model = keras.Sequential()
        # # self.model = ResNet50()
        # self.model.add(keras.layers.ZeroPadding2D((l_r_padding, u_d_padding), input_shape=(box[3] - box[1], box[4] - box[2], 3)))
        # self.model.add(ResNet50())

        # self.model = ResNet50(weights='imagenet')

        # crop_img = cv2.copyMakeBorder(crop_img, u_d_padding, u_d_padding, l_r_padding, l_r_padding, cv2.BORDER_CONSTANT, value=[255, 255, 255])
        # cv2.imshow("cropped", crop_img)

        cv2.imwrite("temp.jpg", crop_img)

        img = image.load_img("temp.jpg", target_size=(224, 224))

        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        # extract features
        with graph.as_default():
            scores = self.model.predict(x)

        sim_class = np.argmax(scores)
        y = decode_predictions(scores, top=1)
        if y[0][0][2] > 0.50:
            return True
        return False

    def round2zero(self, val):
        if val < 0:
            return 0
        else:
            return val

    def get_box_imges_from_img(self, img, offset_height, offset_width, target_height, target_width):
        img_box = tf.image.crop_to_bounding_box(img, offset_height, offset_width, target_height, target_width)
        return img_box