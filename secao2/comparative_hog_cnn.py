# -*- coding: utf-8 -*-

import cv2
import dlib

#image = cv2.imread('../materials/fotos/grupo.2.jpg')

faces = []

for i in range(9):
    image = cv2.imread('../materials/fotos/grupo.' + str(i) + '.jpg')
    faces.append(image)

# HOG
detector_hog = dlib.get_frontal_face_detector()
detected_faces_hog, score, idx = detector_hog.run(faces[0], 1)

# CNN
detector_CNN = dlib.cnn_face_detection_model_v1('../materials/recursos/mmod_human_face_detector.dat')
detected_faces_CNN = detector_CNN(faces[0], 1)


for i, d in enumerate(detected_faces_hog):
    print(score[i])

print()

for face in detected_faces_CNN:
    print(face.confidence)

