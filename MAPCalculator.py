import cnn_utils as utils
from cocoApi.pycocotools.coco import COCO

class MAPCalculator :
    def __init__(self):
        self.annFile = annFile='/home/dima/YoloAndmRCNN/annonations_json/instances_val2014.json'
        self.coco = COCO(annFile)

    def imageNameToId(self, image_name):
        return int(image_name[13:25])

    def getGroundTruth(self,image_name):
        ground_truth = []
        image_id = self.imageNameToId(image_name)

        ann_ids = self.coco.getAnnIds(image_id)
        image_anns = self.coco.loadAnns(ann_ids)

        for ann in image_anns:
            category = ann[0]
            x = ann[1]
            y = ann[2]
            h = ann[3]
            w = ann[4]
            ground_truth.append([category,x, y, x + w, y + h])

        return ground_truth

    def savePredictionsToFile(self,path,predictions):
        with open(path, 'w+') as filehandle:
            for line in predictions:
                filehandle.write( ' '.join([str(x) for x in line]))
                filehandle.write('\n')


calc = MAPCalculator()
calc.savePredictionsToFile("test.txt",[["dima",1,2,3,4],["adi",3,4,5,6]])