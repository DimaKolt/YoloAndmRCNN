
from cocoApi.pycocotools.coco import COCO
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import pylab

pylab.rcParams['figure.figsize'] = (8.0, 10.0)

dataDir='..'
dataType='trainval2014'
annFile='/home/dima/YoloAndmRCNN/annonations_json/instances_val2014.json'

# initialize COCO api for instance annotations
coco=COCO(annFile)

# display COCO categories and supercategories
# cats = coco.loadCats(coco.getCatIds())
# nms=[cat['name'] for cat in cats]
# print('COCO categories: \n{}\n'.format(' '.join(nms)))
imgIds=sorted(coco.getImgIds(imgIds = [324158]))
# imgIds=imgIds[0:100]
# imgId = imgIds[np.random.randint(100)]

id = coco.getAnnIds(imgIds)
imageAnns = coco.loadAnns(id)
image_name_str = coco.imgs[imgIds[0]]['file_name']
#'COCO_val2014_000000391895.jpg'

image = coco.loadImgs(id)

IDs = coco.getImgIds([1])


