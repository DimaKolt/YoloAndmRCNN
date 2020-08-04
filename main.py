import combCNN.CombinedNet as my_net

chosen_pic = "/home/dima/YoloAndmRCNN/images/4410436637_7b0ca36ee7_z.jpg"

combinedNet = my_net()
combinedNet.readImage(chosen_pic)
res = combinedNet.filterByIOU(0.9)

combinedNet.show(res)

coco = 7
