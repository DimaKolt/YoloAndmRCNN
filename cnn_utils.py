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


class ResultFilter:
    def __init__(self, yolo_pred, mrcnn_pred):
        self.yolo_pred = yolo_pred.copy()
        self.mrcnn_pred = self.convert_to_yolo_pred(mrcnn_pred)



    def convert_to_yolo_pred(self, mrcnn_pred):
        rois = mrcnn_pred['rois']
        class_ids = mrcnn_pred['class_ids']
        scores = mrcnn_pred['scores']
        class_names = mrcnn_pred['class_names']
        ress = []
        for i in range(len(rois)):
            ress.append(  # TODO class ids -1
                [class_names[i], class_ids[i], scores[i], round(rois[i][1]), round(rois[i][0]), round(rois[i][3]),
                 round(rois[i][2])])
        return ress


    def filter_IOU_and_score(self, iou_bar):
        yolo_pred_temp = self.yolo_pred.copy()
        mr_pred_temp = self.mrcnn_pred.copy()

        for obj_yolo in self.yolo_pred:
            for obj_mr in self.mrcnn_pred:
                box_yolo = [obj_yolo[3], obj_yolo[4], obj_yolo[5], obj_yolo[6]]
                box_mr = [obj_mr[3], obj_mr[4], obj_mr[5], obj_mr[6]]
                iou = bb_intersection_over_union(box_yolo, box_mr)
                if iou >= iou_bar:
                    if obj_yolo[2] >= obj_mr[2]:
                        mr_pred_temp.remove(obj_mr)
                    else:
                        yolo_pred_temp.remove(obj_yolo)
                    break

        return yolo_pred_temp + mr_pred_temp







