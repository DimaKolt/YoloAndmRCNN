import cnn_utils as utils
from cocoApi.pycocotools.coco import COCO


class MAPCalculator:
    def __init__(self):
        self.annFile = '/home/dima/YoloAndmRCNN/annonations_json/instances_train2014.json'
        self.coco = COCO(self.annFile)

    def imageNameToId(self, image_name):
        return int(image_name[-16:-4])

    def getGroundTruth(self, image_name):
        ground_truth = []
        image_id = self.imageNameToId(image_name)

        ann_ids = self.coco.getAnnIds(image_id)
        image_anns = self.coco.loadAnns(ann_ids)

        for ann in image_anns:
            category_id = ann['category_id']
            cat_list = self.coco.dataset['categories']
            cat_name = ''
            for i in cat_list:
                if i['id']==category_id:
                    cat_name = i['name']
                    break
            bbox = ann['bbox']
            x = bbox[0]
            y = bbox[1]
            w = bbox[2]
            h = bbox[3]
            ground_truth.append([cat_name, x, y, x + w, y + h])

        return ground_truth




