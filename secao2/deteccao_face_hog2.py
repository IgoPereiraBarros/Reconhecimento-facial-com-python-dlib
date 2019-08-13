# -*- coding: utf-8 -*-

import cv2
import dlib
import logging

logger = logging.getLogger(__name__)

subdetector = ['Olhar a frente', 'Vista a esquerda', 'Vista a direita',
               'A frente girando a esquerda', 'A frente girando a direita']

image = cv2.imread('../materials/fotos/grupo.0.jpg')

detector = dlib.get_frontal_face_detector()
detected_faces, score, idx = detector.run(image)

#logger.warning(detected_faces)
#logger.warning(score)
#logger.warning(idx)

for i, d in enumerate(detected_faces):
    logger.warning(f'Detecçao: {d}, Pontuação: {score[i]}, Sub-detector: {subdetector[idx[i]]}')
    l, t, r, b = int(d.left()), int(d.top()), int(d.right()), int(d.bottom())
    cv2.rectangle(image, (l, t), (r, b), (231, 122, 211), 2)

cv2.imshow('Faces detectadas', image)
cv2.waitKey(0)
cv2.destroyAllWindows()