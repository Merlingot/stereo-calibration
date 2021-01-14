from modules.StereoCalibration import StereoCalibration
from modules.util import *

# ================ PARAMETRES ========================

patternSize=(15,10)
squaresize=7e-2
# single_path='captures_zed/captures_calibration_int/'
stereo_path='captures_zed/captures_test_distances/'
single_detected_path='output/singles_detected/'
stereo_detected_path='output/stereo_detected/'
# ====================================================

obj = StereoCalibration(patternSize, squaresize)
# obj.calibrateSingle(single_path, single_detected_path, fisheye=True)
obj.calibrateStereo(stereo_path, stereo_detected_path, single_detected_path, fisheye=False, calib_2_sets=False)
# single_path=single_path
obj.saveResultsXML()
