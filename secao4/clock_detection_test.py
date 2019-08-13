# -*- coding: utf-8 -*-

import os
from glob import glob

import cv2
import dlib

print(dlib.test_simple_object_detector('../materials/recursos/test_clock.xml', '../materials/recursos/clock_detector.svm'))

clock_detector = dlib.simple_object_detector('../materials/recursos/clock_detector.svm')
for image in glob(os.path.join('../materials/relogios_teste', '*.jpg')):
    img = cv2.imread(image)
    detected_objects = clock_detector(img, 2)
    
    for _object in detected_objects:
        l, t, r, b = int(_object.left()), int(_object.top()), int(_object.right()), int(_object.bottom())
        cv2.rectangle(img, (l, t), (r, b), (0, 255, 255), 2)
        
    cv2.imshow('Relógios detectados', img)
    cv2.waitKey(0)

cv2.destroyAllWindows()