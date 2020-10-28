import cv2
import numpy as np

class YOLONet:

    def __init__(self):
        self.scale = 0.00392
        self.classes = None
        with open("./YOLO_env/yolov3.txt", 'r') as f:
            self.classes = [line.strip() for line in f.readlines()]

        self.COLORS = np.random.uniform(0, 255, size=(len(self.classes), 3))
        self.net = cv2.dnn.readNet("./YOLO_env/yolov3.weights", "./YOLO_env/yolov3.cfg") #TODO fix paths
        self.dictionary = dict(zip(self.classes, self.COLORS))


    def __get_output_layers(self,net):
        layer_names = net.getLayerNames()

        output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

        return output_layers

    def __draw_prediction(self, img, class_str, confidence, x, y, x_plus_w, y_plus_h, alg=''):
        label = class_str

        color = self.dictionary[label]

        cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 2)

        cv2.putText(img, label +' '+ alg, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    def readImage(self,image_path):
        self.image_path = image_path;
        self.image = cv2.imread(image_path)
        self.Width = self.image.shape[1]
        self.Height = self.image.shape[0]
        blob = cv2.dnn.blobFromImage(self.image, self.scale, (416, 416), (0, 0, 0), True, crop=False)
        self.net.setInput(blob)

    def predict(self):
        outs = self.net.forward(self.__get_output_layers(self.net))

        class_ids = []
        confidences = []
        boxes = []
        conf_threshold = 0.5
        nms_threshold = 0.4

        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    center_x = int(detection[0] * self.Width)
                    center_y = int(detection[1] * self.Height)
                    w = int(detection[2] * self.Width)
                    h = int(detection[3] * self.Height)
                    x = center_x - w / 2
                    y = center_y - h / 2
                    class_ids.append(class_id)
                    confidences.append(float(confidence))
                    boxes.append([x, y, w, h])

        indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
        results = []

        for i in indices:
            i = i[0]
            box = boxes[i]
            x = box[0]
            y = box[1]
            w = box[2]
            h = box[3]
            # results.append([class_ids[i], confidences[i], round(x), round(y), round(x + w), round(y + h)])
            results.append(
                [str(self.classes[int(class_ids[i])]), confidences[i], round(x), round(y), round(x + w), round(y + h), "YOLO"])
            # draw_prediction(image, class_ids[i], confidences[i], round(x), round(y), round(x + w), round(y + h))
        return results

    def show(self, res):
        for result in res:
            # class_str, confidence, x, y, x_plus_w, y_plus_h, alg = result
            class_str, confidence, x, y, x_plus_w, y_plus_h = result
            # self.__draw_prediction(self.image, class_str, confidence, x, y, x_plus_w , y_plus_h, alg)
            self.__draw_prediction(self.image, class_str, confidence, x, y, x_plus_w, y_plus_h)

        # cv2.imshow("object detection", self.image)
        # cv2.waitKey()

        cv2.imwrite("temp.jpg", self.image)
        self.image = cv2.imread(self.image_path)
        # cv2.destroyAllWindows()