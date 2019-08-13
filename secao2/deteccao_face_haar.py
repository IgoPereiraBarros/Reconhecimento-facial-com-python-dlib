# -*- coding: utf-8 -*-

import cv2



image = cv2.imread('../materials/fotos/grupo.0.jpg')
classifier = cv2.CascadeClassifier('../materials/recursos/haarcascade_frontalface_default.xml')

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

detected_faces = classifier.detectMultiScale(gray, scaleFactor=1.2, minSize=(50, 50))

for (x, y, w, h) in detected_faces:
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

cv2.imshow('Faces detectadas', image)
cv2.waitKey(0)

cv2.destroyAllWindows()