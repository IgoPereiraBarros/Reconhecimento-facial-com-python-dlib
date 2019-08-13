# -*- coding: utf-8 -*-

from glob import glob
import logging
import os
from PIL import Image
import _pickle as cPickle

import cv2
import dlib
import numpy as np


logger = logging.getLogger(__name__)

face_detector = dlib.get_frontal_face_detector()
point_detectors = dlib.shape_predictor('../materials/recursos/shape_predictor_68_face_landmarks.dat')
facial_recognition = dlib.face_recognition_model_v1('../materials/recursos/dlib_face_recognition_resnet_model_v1.dat')

indexes = np.load('../materials/recursos/indexes_yales.pickle')
facial_descriptors = np.load('../materials/recursos/descriptors_yale.npy')

threshold = 0.5
total_faces = 0
total_hits = 0

for file in glob(os.path.join('../materials/yalefaces/teste', '*.gif')):
    face_image = Image.open(file).convert('RGB')
    image = np.asarray(face_image, 'uint8')
    
    current_id = int(os.path.split(file)[1].split('.')[0].replace('subject', ''))
    total_faces += 1
    
    detected_faces = face_detector(image, 2)
    
    for face in detected_faces:
        f, t, r, b = int(face.left()), int(face.top()), int(face.right()), int(face.bottom())
        facial_points = point_detectors(image, face)
        facial_descriptor = facial_recognition.compute_face_descriptor(image, facial_points)
        
        list_facial_descriptor = np.array([fd for fd in facial_descriptor], dtype=np.float64)
        list_facial_descriptor = list_facial_descriptor[np.newaxis, :]
        
        distances = np.linalg.norm(list_facial_descriptor - facial_descriptors, axis=1)
        minimum = np.argmin(distances)
        min_distance = distances[minimum]
        
        if min_distance <= threshold:
            name = os.path.split(indexes[minimum])[1].split('.')[0]
            predict_id = int(os.path.split(indexes[minimum])[1].split('.')[0].replace('subject', ''))
            if predict_id == current_id:
                total_hits += 1
        else:
            name = 'Desconhecido'
        
        logger.warning('ID atual: {}, ID previsto: {}'.format(current_id, predict_id))
        cv2.rectangle(image, (f, t), (r, b), (0, 0, 255), 2)
        text = '{} {:.4f}'.format(name, min_distance)
        cv2.putText(image, text, (r, t), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5, (0, 0, 255))
    
    cv2.imshow('Faces reconhecidas', image)
    cv2.waitKey(0)

percent_hits = (total_hits / total_faces) * 100
logger.warning('Percentual de acertos: {}'.format(percent_hits))
cv2.destroyAllWindows()