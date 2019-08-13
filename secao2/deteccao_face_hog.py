# -*- coding: utf-8 -*-

import cv2
import dlib



image = cv2.imread('../materials/fotos/grupo.0.jpg')
detector = dlib.get_frontal_face_detector()

detected_faces = detector(image, 1)

for face in detected_faces:
    l, t, r, b = (int(face.left()), int(face.top()), int(face.right()), int(face.bottom()))
    cv2.rectangle(image, (l, t), (r, b), (255, 0, 0), 2)

cv2.imshow('Faces detectadas', image)
cv2.waitKey(0)

cv2.destroyAllWindows()