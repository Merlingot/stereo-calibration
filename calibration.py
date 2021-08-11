from modules.StereoCalibration import StereoCalibration
from modules.util import *
import cv2 as cv
import matplotlib.pyplot as plt

################################################################################
# Paramètres du damier
patternSize=(8,10) #Patron du damier
squaresize=2e-2 #Taille d'un carreau du damier


# Path vers les images à analyser
single_path='captures/captures_calibration/'
stereo_path='captures/captures_calibration/'


# Path vers les folders où enregistrer les images détectées
single_detected_path='output/singles_detected/' #images détectée lors de la calibration individuelle
stereo_detected_path='output/stereo_detected/'#images détectée lors de la calibration stéréo
################################################################################


obj = StereoCalibration(patternSize, squaresize)
obj.calibrateSingle(single_path, single_detected_path, fisheye=False, draw=False)
obj.draw=True
obj.calibrateStereo(stereo_path, stereo_detected_path, fisheye=False, calib_2_sets=False, draw=False)
obj.saveResultsXML(left_name='cam1', right_name='cam2')
obj.perViewErrors
obj.reprojection('output/reprojection/',5)
