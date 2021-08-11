from modules.StereoCalibration import StereoCalibration
from modules.util import *
import cv2 as cv
import matplotlib.pyplot as plt

from config import * #importation des paramètres

obj = StereoCalibration(patternSize, squareSize)
obj.calibrateSingle(single_path, single_detected_path, fisheye=False, draw=False)
obj.draw=True
obj.calibrateStereo(stereo_path, stereo_detected_path, fisheye=False, calib_2_sets=False, draw=False)
obj.saveResultsXML(left_name='cam1', right_name='cam2')
obj.perViewErrors
obj.reprojection('output/reprojection/',5)
