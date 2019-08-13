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

index = {}
idx = 0
facial_descriptors = None

for file in glob(os.path.join('../materials/yalefaces/treinamento', '*.gif')):
    face_image = Image.open(file).convert('RGB')
    image = np.array(face_image, 'uint8')
    
    detected_faces = face_detector(image, 1)
    
    number_detected_faces = len(detected_faces)
    if number_detected_faces > 1:
        logger.warning(f'Há mais de uma face nesta imagem {file}')
        exit(0)
    elif number_detected_faces < 1:
        logger.warning(f'Nenhuma face encontrada na imagem {file}')
        exit(0)
    
    for face in detected_faces:
        facial_points = point_detectors(image, face)
        facial_descriptor = facial_recognition.compute_face_descriptor(image, facial_points)
        
        list_facial_descriptors = np.array([fd for fd in facial_descriptor], dtype=np.float64)
        
        list_facial_descriptors = list_facial_descriptors[np.newaxis, :]
        
        if facial_descriptors is None:
            facial_descriptors = list_facial_descriptors
        else:
            facial_descriptors = np.concatenate((facial_descriptors, list_facial_descriptors), axis=0)
        
        index[idx] = file
        idx += 1
    #cv2.imshow('Face detectada', image)
    #cv2.waitKey(0)

np.save('../materials/recursos/descriptors_yale.npy', facial_descriptors)
with open('../materials/recursos/indexes_yales.pickle', 'wb') as f:
    cPickle.dump(index, f)

#cv2.destroyAllWindows()
