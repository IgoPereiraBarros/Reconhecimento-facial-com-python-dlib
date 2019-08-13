# -*- coding: utf-8 -*-

import dlib
import cv2
import numpy as np


def print_points(image, facial_points):
    for f in facial_points.parts():
        cv2.circle(image, (f.x, f.y), 2, (0, 255, 0), 2)


face_detector = dlib.get_frontal_face_detector()
detector_points = dlib.shape_predictor('../materials/recursos/shape_predictor_5_face_landmarks.dat')

image = cv2.imread('../materials/fotos/treinamento/ronald.0.1.jpg')
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

detected_faces = face_detector(image_rgb)
face_points = dlib.full_object_detections()

for face in detected_faces:
    points = detector_points(image_rgb, face)
    face_points.append(points)
    print_points(image, points)

images = dlib.get_face_chips(image_rgb, face_points)

for img in images:
    image_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imshow('Imagem original', image)
    cv2.waitKey(0)
    cv2.imshow('Imagem alinhada', image_bgr)
    cv2.waitKey(0)

#cv2.imshow('5 pontos', image)
#cv2.waitKey(0)
cv2.destroyAllWindows()
