import numpy as np
import cv2
import configparser
import math
import argparse



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
    return width,height
##############################################################################
