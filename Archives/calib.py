from StereoCalibration import StereoCalibration
import glob
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from util import *

# ================ PARAMETRES ========================

patternSize=(10,8)
squaresize=2e-2
# single_path='stereo/'
stereo_path='captures/captures_calibration/'
single_path=stereo_path
# single_detected_path='output/singles_detected/'
stereo_detected_path='output/stereo_detected/'
single_detected_path=stereo_detected_path
# ====================================================

# INFO UTILES ------------------------------------------------------------------
objp = coins_damier(patternSize,squaresize)
# Critères
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
not_fisheye_flags = cv.CALIB_FIX_K3|cv.CALIB_ZERO_TANGENT_DIST
# ------------------------------------------------------------------------------



# LECTURE IMAGE ET DÉTECTION
images_left = np.sort(glob.glob(stereo_path + 'left*.jpg'))
images_right = np.sort(glob.glob(stereo_path + 'right*.jpg'))

# Parcours le dossier pour trouver le damier sur les images
objpoints_l=[]; objpoints_r=[]; imgpoints_l=[]; imgpoints_r=[]
objpoints=[]; imgpoints_left=[]; imgpoints_right=[]
for i in range(len(images_right)):
    gray_l = cv.cvtColor( cv.imread(images_left[i]), cv.COLOR_RGB2GRAY)
    ret_l, corners_l = cv.findChessboardCorners(gray_l, patternSize, None)
    gray_r = cv.cvtColor( cv.imread(images_right[i]), cv.COLOR_RGB2GRAY)
    ret_r, corners_r = cv.findChessboardCorners(gray_r, patternSize, None)

    if ret_l:
        objpoints_l.append(objp)
        corners2= cv.cornerSubPix(gray_l, corners_l, (11, 11),(-1, -1), criteria)
        imgpoints_l.append(corners2)
    if ret_r:
        objpoints_r.append(objp)
        corners2= cv.cornerSubPix(gray_r, corners_r, (11, 11),(-1, -1), criteria)
        imgpoints_r.append(corners2)
    # Si le damier dans les images gauche et droite correspondante est détecté
    if ret_l*ret_r==1:
        objpoints.append(objp)
        corners2= cv.cornerSubPix(gray_l, corners_l, (11, 11),(-1, -1), criteria)
        imgpoints_left.append(corners2)
        corners2= cv.cornerSubPix(gray_r, corners_r, (11, 11),(-1, -1), criteria)
        imgpoints_right.append(corners2)
imageSize1=imageSize2=gray_l.shape[:2]

# CALIBRATION INDIVIDUELLE -----------------------------------------------------
err1, M1, d1, r1, t1, stdDeviationsIntrinsics1, stdDeviationsExtrinsics1, perViewErrors1 = cv.calibrateCameraExtended(objpoints_l, imgpoints_l, imageSize1, None, None)

err2, M2, d2, r2, t2, stdDeviationsIntrinsics2, stdDeviationsExtrinsics2, perViewErrors2 = cv.calibrateCameraExtended(objpoints_r, imgpoints_r, imageSize2, None, None, flags=not_fisheye_flags)
# ------------------------------------------------------------------------------

# CALIBRATION STEREO ---------------------------------------------------------
# flags=cv.CALIB_FIX_K3|cv.CALIB_ZERO_TANGENT_DIST
errStereo, _, _, _, _, R, T, E, F, stereo_per_view_err= cv.stereoCalibrateExtended(objpoints, imgpoints_left, imgpoints_right, M1, d1, M2,d2, imageSize1, None, None, flags=not_fisheye_flags)

# Enlever les outliers et recalibrer:
indices=np.indices(stereo_per_view_err.shape)[0]
indexes=indices[stereo_per_view_err>errStereo*2]
if len(indexes)>0:
    for i in indexes:
        objpoints.pop(i)
        imgpoints_left.pop(i)
        imgpoints_right.pop(i)
    # re-calculs
    errStereo, _, _, _, _, R, T, E, F, stereo_per_view_err= cv.stereoCalibrateExtended(objpoints, imgpoints_left, imgpoints_right, M1, d1, M2,d2, imageSize1, None, None,flags=not_fisheye_flags)
