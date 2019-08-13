# -*- coding: utf-8 -*-

from glob import glob
import os

import dlib
import cv2


detector = dlib.simple_object_detector('../materials/recursos/clock_detector.svm')
detector_points = dlib.shape_predictor('../materials/recursos/detector_clock_points.dat')

print(dlib.test_shape_predictor('../materials/recursos/clock_points_test.xml', '../materials/recursos/detector_clock_points.dat'))

def print_points(image, points):
    for p in points.parts():
        cv2.circle(image, (p.x, p.y), 2, (255, 0, 0), 3)
        

for file in glob(os.path.join('../materials/relogios_teste', '*.jpg')):
    image = cv2.imread(file)
    detected_objects = detector(image, 2)
    
    for _object in detected_objects:
        l, t, r, b = int(_object.left()), int(_object.top()), int(_object.right()), int(_object.bottom())
        cv2.rectangle(image, (l, t), (r, b), (0, 0, 255), 2)
        points = detector_points(image, _object)
        print_points(image, points)
    
    cv2.imshow('Detector points', image)
    cv2.waitKey(0)

cv2.destroyAllWindows()