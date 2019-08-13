# -*- coding: utf-8 -*-

import logging

import cv2
import dlib

logger = logging.getLogger(__name__)

webcam = cv2.VideoCapture(0)

image = cv2.imread('../materials/fotos/grupo.0.jpg')
detector = dlib.cnn_face_detection_model_v1('../materials/recursos/mmod_human_face_detector.dat')

connected, frame = webcam.read()

detected_faces = detector(frame, 1)

logger.warning(detected_faces)

for face in detected_faces:
    '''
        l --> left
        t --> top
        r --> right
        b --> bottom
        c --> confidence
    '''
    
    l, t, r, b, c = (int(face.rect.left()), int(face.rect.top()), int(face.rect.right()), \
                    int(face.rect.bottom()), face.confidence)
    logger.warning(c)
    cv2.rectangle(frame, (l, t), (r, b), (255, 0, 0), 2)

cv2.imshow('Faces detectadas', frame)
cv2.waitKey(0)
cv2.destroyAllWindows()