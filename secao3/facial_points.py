# -*- coding: utf-8 -*-

import cv2
import dlib
import numpy as np


font = cv2.FONT_HERSHEY_COMPLEX_SMALL

def print_dots(image, facial_points):
    for point in facial_points.parts():
        cv2.circle(image, (point.x, point.y), 2, (0, 255, 0), 2)

def print_numbers(image, facial_points):
    for i, p in enumerate(facial_points.parts()):
        cv2.putText(image, str(i), (p.x, p.y), font, .55, (0, 0, 255), 1)

def print_lines(image, facial_points):
    p68 = [[0, 16, False], # linha do queixo
           [17, 21, False], # sombrancelha direita
           [22, 26, False], # sombrancelha esquerda
           [27, 30, False], # ponte nasal
           [30, 35, True], # nariz inferior
           [36, 41, True], # olho esquerdo
           [42, 47, True], # olho direito
           [48, 59, True], # lábio externo
           [60, 67, True]] # lábio interno
    
    for k in range(len(p68)):
        points = []
        for i in range(p68[k][0], p68[k][1] + 1):
            point = [facial_points.part(i).x, facial_points.part(i).y]
            points.append(point)
        points = np.array(points, dtype=np.int32)
        cv2.polylines(image, [points], p68[k][2], (0, 0, 255), 2)

'''
def return_faces(id_person):
    faces = []
    if id_person == 0:
        for i in range(5):
            image = cv2.imread('../materials/fotos/treinamento/ronald.' + str(id_person) + '.' + str(i) + '.jpg')
            faces.append(image)
    elif id_person == 1:
        for i in range(3):
            image = cv2.imread('../materials/fotos/treinamento/nancy.' + str(id_person) + '.' + str(i) + '.jpg')
            faces.append(image)

    return np.array(faces)
'''

#image = cv2.imread('../materials/fotos/treinamento/ronald.0.1.jpg')
image = cv2.imread('../materials/fotos/grupo.5.jpg')

face_detector = dlib.get_frontal_face_detector()
points_detectors = dlib.shape_predictor('../materials/recursos/shape_predictor_68_face_landmarks.dat')
detected_faces = face_detector(image, 2)


for face in detected_faces:
    points = points_detectors(image, face)
    #print_dots(image, points)
    print_numbers(image, points)
    #print_lines(image, points)


cv2.imshow('Faces reconhecidas', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
