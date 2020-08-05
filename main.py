from combCNN import CombinedNet as my_net

pics_path = "/home/dima/Downloads/train2014/"
ground_truth_path = "/home/dima/YoloAndmRCNN/groundtruth/"
prediction_path = "/home/dima/YoloAndmRCNN/prediction/"
chosen_pic = ""

combinedNet = my_net()
combinedNet.readImage(chosen_pic)
combinedNet.predict()
res = combinedNet.filterByIOU(0.8)

combinedNet.show(res)

coco = 7
