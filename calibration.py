from modules.StereoCalibration import StereoCalibration
from modules.util import *
import cv2 as cv

# ================ PARAMETRES ========================
patternSize=(15,10)
squaresize=7e-2
single_path='zed/tout/'
stereo_path='zed/tout/'
single_detected_path='output/singles_detected/'
stereo_detected_path='output/stereo_detected/'
# ====================================================


cibles=np.genfromtxt("zed/objpts.txt").astype(np.float32)
cibles_l=np.genfromtxt("zed/pts_left.txt").astype(np.float32)
cibles_l=cibles_l.reshape(cibles_l.shape[0], 1, 2)
cibles_r=np.genfromtxt("zed/pts_right.txt").astype(np.float32)
cibles_r=cibles_r.reshape(cibles_r.shape[0], 1, 2)
cibles=cibles[:cibles_r.shape[0], :]


obj = StereoCalibration(patternSize, squaresize)
obj.calibrateIntrinsics(single_path, single_detected_path, cibles=cibles, cibles_l=cibles_l, cibles_r=cibles_r, fisheye=False)
obj.calibrateExtrinsics(stereo_path, stereo_detected_path, cibles=cibles, cibles_l=cibles_l, cibles_r=cibles_r, fisheye=False)
obj.saveResultsXML(left_name='cam1_cibles', right_name='cam2_cibles')
# obj.reprojection('output/reprojection/')







# CAMÉRA GAUCHE
# fx=528.915;fy=528.67
# cx=618.42;cy=372.7495
# k1=-0.0367561;k2=0.00561791;k3=-0.00340223
# p1=-0.000624132;p2=9.12931e-05
# obj.M1 = np.array([[fx,0,cx],[0,fy,cy],[0,0,1]])
# obj.d1 = np.array([k1,k2,p1,p2,k3])
# obj.imageSize1=(720, 1280)
# # CAMÉRA DE DROITE
# fx=527.625; fy=527.465
# cx=655.01;cy=362.3045
# k1=-0.0371171;k2=0.00457376; k3=-0.0029894
# p1=-0.000213958; p2=0.000327238
# obj.imageSize2=(720, 1280)
# obj.M2 = np.array([[fx,0,cx],[0,fy,cy],[0,0,1]])
# obj.d2 = np.array([k1,k2,p1,p2,k3])
# # EXTRINSÈQUES
# RX_HD=-0.000130076
# CV_HD=-0.0090896
# RZ_HD=-0.000337072
# angles=np.array([RX_HD, CV_HD, RZ_HD])
# obj.R, _ = cv.Rodrigues(angles)
# TX=-119.728e-3
# TY=0.0288645e-3
# TZ=-0.613716e-3
# obj.T=np.array([TX, TY, TZ])
