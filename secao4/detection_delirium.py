# -*- coding: utf-8 -*-

from glob import glob
import os

import cv2
import dlib


#options = dlib.simple_object_detector_training_options()
#options.add_left_right_image_flips = True
#options.C = 5

#dlib.train_simple_object_detector('../materials/recursos/training_delirium.xml', '../materials/recursos/detectation_delirium.svm', options)

detector = dlib.simple_object_detector('../materials/recursos/detectation_delirium.svm')

for file in glob(os.path.join('../materials/delirium', '*.jpg')):
    image = cv2.imread(file)
    detected_objects = detector(image, 2)
    
    for _object in detected_objects:
        l, t, r, b = int(_object.left()), int(_object.top()), int(_object.right()), int(_object.bottom())
        cv2.rectangle(image, (l, t), (r, b), (0, 0, 255), 2)
    
    cv2.imshow('Detector delirium', image)
    cv2.waitKey(0)

cv2.destroyAllWindows()