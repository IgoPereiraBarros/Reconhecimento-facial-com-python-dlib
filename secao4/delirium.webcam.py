# -*- coding: utf-8 -*-

import sys

import cv2
import dlib


step_squares = 30
captura = cv2.VideoCapture(0)
count_squares = 0
detector = dlib.simple_object_detector('../materials/recursos/detectation_delirium.svm')


while captura.isOpened():
    connected, frame = captura.read()
    count_squares += 1
    if count_squares % step_squares == 0:
        detected_objects = detector(frame, 1)
        for _object in detected_objects:
            l, t, r, b = int(_object.left()), int(_object.top()), int(_object.right()), int(_object.bottom())
            cv2.rectangle(frame, (l, t), (r, b), (255, 0, 0), 2)
        cv2.imshow('Delirium', frame)
        
        if cv2.waitKey(1) == ord('c'):
            break

captura.release()
cv2.destroyAllWindows()
sys.exit(0)
