import numpy as np
import cv2

import math

from util import readXML



##############################################################################
def get_image_dimension_from_resolution(resolution):

    if resolution== 'VGA' :
 #       width=672
#        height=376
        width=800
        height=480
    elif resolution== 'HD' :
        width=1280
        height=720
    elif resolution== 'FHD' :
        width=1920
        height=1080
    elif resolution== '2K' :
        width=2208
        height=1242
    return width,height
##############################################################################



##############################################################################
# THIS FONCTION READS CALIBRATION FILE AND OUTPUTS MATRICES AND COEFFICIENTS
## SEE EXAMPLE SN20716499.conf FOR CORRECT FORMAT

def init_calibration(left_xml, right_xml, image_size, resolution) :


    # READ .XML FILE -------
    cameraMatrix_left,distCoeffs_left, _, _ ,_, E, F = readXML(left_xml) # left
    cameraMatrix_right,distCoeffs_right, R, T ,_, _, _ = readXML(right_xml) # right

    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(cameraMatrix1=cameraMatrix_left,
                                       cameraMatrix2=cameraMatrix_right,
                                       distCoeffs1=distCoeffs_left,
                                       distCoeffs2=distCoeffs_right,
                                       R=R, T=T,
                                       flags=cv2.CALIB_ZERO_DISPARITY,
                                       alpha=0,
                                       imageSize=(image_size.width, image_size.height),
                                       newImageSize=(image_size.width, image_size.height))

    map_left_x, map_left_y = cv2.initUndistortRectifyMap(cameraMatrix_left, distCoeffs_left, R1, P1, (image_size.width, image_size.height), cv2.CV_32FC1)
    map_right_x, map_right_y = cv2.initUndistortRectifyMap(cameraMatrix_right, distCoeffs_right, R2, P2, (image_size.width, image_size.height), cv2.CV_32FC1)

    cameraMatrix_left = P1
    cameraMatrix_right = P2

    return cameraMatrix_left, cameraMatrix_right, map_left_x, map_left_y, map_right_x, map_right_y, Q

###############################################################################






################################################################################
## THIS FONCTION RECTIFIES LEFT AND RIGHT FRAMES

def get_rectified_left_right(left_frame,right_frame, map_left_x, map_left_y, map_right_x, map_right_y):

    left_frame_rect = cv2.remap(left_frame, map_left_x, map_left_y, interpolation=cv2.INTER_LINEAR)
    right_frame_rect = cv2.remap(right_frame, map_right_x, map_right_y, interpolation=cv2.INTER_LINEAR)

    return left_frame_rect,right_frame_rect
##############################################################################
