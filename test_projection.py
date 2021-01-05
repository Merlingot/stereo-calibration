from modules.util import *
from modules.points3d import *

# Fichiers de calibration:
left_xml='cam1.xml'
right_xml='cam2.xml'
cam1, cam2 = get_cameras(left_xml, right_xml)

patternSize=(10,8)
squaresize=2e-2
folder='captures'

world_th=coins_damier(patternSize,squaresize).T

# RESOLUTION ET TAILLE -----------------------------------------------------
resolution='VGA'
width,height = get_image_dimension_from_resolution(resolution)
image_size = Resolution(width,height)
# --------------------------------------------------------------------------

# LIRE FICHIERS DE CALIBRATION ---------------------------------------------
K1,d1, _, _ ,_,E, F = readXML(left_xml) # left
K2,d2, R, T ,_, _, _ = readXML(right_xml) # right
# --------------------------------------------------------------------------

# RECTIFICATION ------------------------------------------------------------
R1, R2, P1, P2, Q, _, _ = cv.stereoRectify(cameraMatrix1=K1, cameraMatrix2=K2,distCoeffs1=d1,distCoeffs2=d2,R=R, T=T, flags=0, alpha=1,
imageSize=(image_size.height, image_size.width))

map_left_x, map_left_y = cv.initUndistortRectifyMap(K1, d1, R1, P1, (image_size.width, image_size.height), cv.CV_32FC1)
map_right_x, map_right_y = cv.initUndistortRectifyMap(K2, d2, R2, P2, (image_size.width, image_size.height), cv.CV_32FC1)
# --------------------------------------------------------------------------

# CAMÉRAS ------------------------------------------------------------------
cam1 = Camera(K1,d1,R1,P1,map_left_x,map_left_y); cam1.Q=Q
cam2 = Camera(K2,d2,R2,P2,map_right_x,map_right_y)
cam1.set_images('{}/captures_calibration/left4.jpg'.format(folder))
cam2.set_images('{}/captures_calibration/right4.jpg'.format(folder))
# --------------------------------------------------------------------------
