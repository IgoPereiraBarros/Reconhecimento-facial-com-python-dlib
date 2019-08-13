# -*- coding: utf-8 -*-

import dlib


options = dlib.shape_predictor_training_options()
dlib.train_shape_predictor('../materials/recursos/training_clock_points.xml', '../materials/recursos/detector_clock_points.dat', options)
