import numpy as np
import cv2 as cv

from util import readXML



##############################################################################
class Resolution :
    def __init__(self,width, height):
        self.width = int(width)
        self.height = int( height)
##############################################################################


##############################################################################
def get_image_dimension_from_resolution(resolution):

    if resolution== 'VGA' :
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
    K1,d1, _, _ ,_, E, F = readXML(left_xml) # left
    K2,d2, R, T ,_, _, _ = readXML(right_xml) # right

    R1, R2, P1, P2, Q, _, _ = cv.stereoRectify(cameraMatrix1=K1,
                                       cameraMatrix2=K2,
                                       distCoeffs1=d1,
                                       distCoeffs2=d2,
                                       R=R, T=T,
                                       flags=0,
                                       alpha=0,
                                       imageSize=(image_size.width, image_size.height),
                                       newImageSize=(image_size.width, image_size.height))

    map_left_x, map_left_y = cv.initUndistortRectifyMap(K1, d1, R1, P1, (image_size.width, image_size.height), cv.CV_32FC1)
    map_right_x, map_right_y = cv.initUndistortRectifyMap(K2, d2, R2, P2, (image_size.width, image_size.height), cv.CV_32FC1)


    return P1, P2, map_left_x, map_left_y, map_right_x, map_right_y, Q, R1, K1, d1

###############################################################################






################################################################################
## THIS FONCTION RECTIFIES LEFT AND RIGHT FRAMES

def get_rectified_left_right(left_frame,right_frame, map_left_x, map_left_y, map_right_x, map_right_y):

    left_frame_rect = cv.remap(left_frame, map_left_x, map_left_y, interpolation=cv.INTER_LINEAR)
    right_frame_rect = cv.remap(right_frame, map_right_x, map_right_y, interpolation=cv.INTER_LINEAR)

    return left_frame_rect,right_frame_rect
##############################################################################
