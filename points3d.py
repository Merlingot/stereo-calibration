import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import glob, os
from stereo_tools import *

from util import readXML, find_corners, read_images, write_ply


# Fichiers de calibration:
left_xml='cam1.xml'
right_xml='cam2.xml'
# Images à reconstruire:
i='2' #numero de l'image
left="stereo/left{}.jpg".format(i)
right="stereo/right{}.jpg".format(i)
resolution='VGA'

# RESOLUTION ET TAILLE ---------------------------------------------------------
width,height = get_image_dimension_from_resolution(resolution)
class Resolution :
    width = int( width)
    height = int( height)
image_size = Resolution()
# ------------------------------------------------------------------------------

# LIRE FICHIERS DE CALIBRATION -------------------------------------------------
P1, P2, map_left_x, map_left_y, map_right_x, map_right_y, Q = init_calibration(left_xml,right_xml,image_size, resolution)
# ------------------------------------------------------------------------------

# RECTIFICATION ----------------------------------------------------------------
frameL=cv2.imread(left)
frameR=cv2.imread(right)
# RECTIFIY
rectifiedL,rectifiedR= get_rectified_left_right(frameL,frameR, map_left_x, map_left_y, map_right_x, map_right_y)
# ------------------------------------------------------------------------------

# STEREO MATCHERS --------------------------------------------------------------
num_disp = 80
min_disp=1
wsize = 3

left_matcher = cv2.StereoSGBM_create(
        minDisparity=min_disp,
        numDisparities = num_disp,
        blockSize=wsize,
        P1=24*wsize*wsize,
        P2=96*wsize*wsize,
        preFilterCap=63,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)
right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)
# ------------------------------------------------------------------------------



# CALCULS DISPARITÉ ------------------------------------------------------------
# GRAY SCALE
grayL = cv2.cvtColor(rectifiedL,cv2.COLOR_BGR2GRAY)
grayR = cv2.cvtColor(rectifiedR,cv2.COLOR_BGR2GRAY)
# DOWNSCALE
downscale=2
new_num_disp = int(num_disp / downscale)
n_width = int(grayL.shape[1] * 1/downscale)
n_height = int(grayR.shape[0] * 1/downscale)
grayL_down = cv2.resize(grayL, (n_width, n_height))
grayR_dowm = cv2.resize(grayR,(n_width, n_height))
# COMPUTE AND FILTER DISPARITY
displ = left_matcher.compute(cv2.UMat(grayL_down),cv2.UMat(grayR_dowm))
dispr = right_matcher.compute(cv2.UMat(grayR_dowm),cv2.UMat(grayL_down))
displ = np.int16(cv2.UMat.get(displ))
dispr = np.int16(cv2.UMat.get(dispr))
# ------------------------------------------------------------------------------

# WLS FILTER ------------------------------------------------------------------
lambd = 8000
sigma = 5
wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=left_matcher)
wls_filter.setLambda(lambd)
wls_filter.setSigmaColor(sigma)
disparity = wls_filter.filter(displ, grayL, None, dispr)
# FORMAT DISPARITY
disparity = disparity.astype(np.float32) / 16.0
plt.imshow(disparity)
#maks
ROI = np.array(wls_filter.getROI())*downscale
conf_map=wls_filter.getConfidenceMap()
mask=np.zeros(conf_map.shape, conf_map.dtype)
mask[ROI[1]:ROI[1]+ROI[3], ROI[0]:ROI[0]+ROI[2]]=1

# ------------------------------------------------------------------------------

# BILATERAL FILTER -------------------------------------------------------------
fbs_spatial=8.0
fbs_luma=8.0
fbs_chroma=8.0
fbs_lambda=128.0
solved_filtered_disp = cv2.ximgproc.fastBilateralSolverFilter(grayL, disparity, conf_map/255.0, None, fbs_spatial, fbs_luma, fbs_chroma, fbs_lambda)

plt.figure()
plt.imshow(solved_filtered_disp)
plt.colorbar()

plt.figure()
plt.imshow(solved_filtered_disp*mask)
plt.colorbar()
# ------------------------------------------------------------------------------


# Reproject to 3d --------------------------------------------
points = cv.reprojectImageTo3D(solved_filtered_disp*mask, Q, handleMissingValues=True)
Z=points[:,:,2]

plt.figure()
plt.imshow(Z*mask)
plt.colorbar()

# Save ----------------------------------------------------
colors = cv.cvtColor(rectifiedL, cv.COLOR_BGR2RGB)
mask=np.zeros(conf_map.shape, dtype=bool)
mask[ROI[1]:ROI[1]+ROI[3], ROI[0]:ROI[0]+ROI[2]]=True

out_points = points[mask]
out_colors = colors[mask]
out_fn = '3dpoints/out_{}.ply'.format(i)
write_ply(out_fn, out_points, out_colors)
