# -*- coding: utf-8 -*-


import cv2
import dlib

#image = cv2.imread('../materials/fotos/grupo.2.jpg')

faces = []

for i in range(9):
    image = cv2.imread('../materials/fotos/grupo.' + str(i) + '.jpg')
    faces.append(image)

font = cv2.FONT_HERSHEY_COMPLEX_SMALL

# HAAR
detector_haar = cv2.CascadeClassifier('../materials/recursos/haarcascade_frontalface_default.xml')
gray = cv2.cvtColor(faces[3], cv2.COLOR_BGR2GRAY)
detected_faces_haar = detector_haar.detectMultiScale(gray, scaleFactor=1.1, minSize=(10, 10))

# HOG
detector_hog = dlib.get_frontal_face_detector()
detected_faces_hog, score, idx = detector_hog.run(faces[3], 1)

# CNN
detector_CNN = dlib.cnn_face_detection_model_v1('../materials/recursos/mmod_human_face_detector.dat')
detected_faces_CNN = detector_CNN(faces[3], 1)

# HAAR
for (x, y, w, h) in detected_faces_haar:
    cv2.rectangle(faces[3], (x, y), (x+w, y+h), (0, 255, 0), 2)
    cv2.putText(faces[3], 'Haar', (x, y - 5), font, 1, (0, 255, 0))

# HOG
for i, d in enumerate(detected_faces_hog):
    l, t, r, b = int(d.left()), int(d.top()), int(d.right()), int(d.bottom())
    cv2.rectangle(faces[3], (l, t), (r, b), (255, 0, 0), 2)
    cv2.putText(faces[3], 'HOG', (r, t), font, 1, (255, 0, 0))

# CNN
for face in detected_faces_CNN:
    '''
        l --> left
        t --> top
        r --> right
        b --> bottom
        c --> confidence
    '''
    l, t, r, b, c = (int(face.rect.left()), int(face.rect.top()), int(face.rect.right()), \
                    int(face.rect.bottom()), face.confidence)
    cv2.rectangle(faces[3], (l, t), (r, b), (0, 0, 255), 2)
    cv2.putText(faces[3], 'CNN', (r, t), font, 1, (0, 0, 255))

cv2.imshow('Comparando os detectores', faces[2])
cv2.waitKey(0)

cv2.destroyAllWindows()