from modules.StereoCalibration import StereoCalibration
from modules.util import *
import cv2 as cv

from config import * #importation des paramètres

obj = StereoCalibration(patternSize, squareSize)
obj.draw=True
obj.calibrateSingle(single_path, single_detected_path, fisheye=False, draw=True)
obj.calibrateStereo(stereo_path, stereo_detected_path, fisheye=False, calib_2_sets=False, draw=True)
obj.saveResultsXML(left_name='cam1', right_name='cam2')
obj.perViewErrors
obj.reprojection('output/reprojection/',5)
