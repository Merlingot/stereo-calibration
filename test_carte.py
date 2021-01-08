from modules.util import *
from modules.points3d import *


# Choisir une image à analyser -------------------------------------------------
fleft = 'captures/captures_zorro/left2.jpg'
fright = 'captures/captures_zorro/right2.jpg'
# ------------------------------------------------------------------------------

# Fichiers de calibration ------------------------------------------------------
left_xml='cam1.xml'
right_xml='cam2.xml'
# ------------------------------------------------------------------------------


cam1,cam2=get_cameras(left_xml, right_xml)
cam1.set_images(fleft)
cam2.set_images(fright)

rectifiedL=cam1.rectified
rectifiedR=cam2.rectified
Q =cam1.Q

# CREATION STEREO MATCHERS -------------------------------------------------
num_disp = 5*16
min_disp= 1
wsize = 5
left_matcher = cv.StereoSGBM_create(
        minDisparity=min_disp,
        numDisparities=num_disp,
        blockSize=wsize,
        P1=24*wsize*wsize,
        P2=96*wsize*wsize,
        preFilterCap=63,
        mode=cv.STEREO_SGBM_MODE_SGBM_3WAY)
right_matcher = cv.ximgproc.createRightMatcher(left_matcher)
# --------------------------------------------------------------------------


# CREATE WLS FILTER --------------------------------------------------------
lambda_wls = 8000.0
sigma_wls = 1.5
wls_filter = cv.ximgproc.createDisparityWLSFilter(matcher_left=left_matcher)
wls_filter.setLambda(lambda_wls)
wls_filter.setSigmaColor(sigma_wls)
# --------------------------------------------------------------------------

# FORMATAGE PRE-CALCUL -----------------------------------------------------
# DOWNSCALE
downscale=2
new_num_disp = int(num_disp / downscale)
n_width = int(rectifiedL.shape[1] * 1/downscale)
n_height = int(rectifiedR.shape[0] * 1/downscale)
downL = cv.resize(rectifiedL, (n_width, n_height))
downR = cv.resize(rectifiedR, (n_width, n_height))

# GRAY SCALE
grayL_down = cv.cvtColor(downL,cv.COLOR_RGB2GRAY)
grayR_down = cv.cvtColor(downR,cv.COLOR_RGB2GRAY)

# SMOOTH
grayL_down = cv.medianBlur(grayL_down,3)
grayR_down = cv.medianBlur(grayR_down,3)
# --------------------------------------------------------------------------


# CALCULS DISPARITÉ --------------------------------------------------------
displ = left_matcher.compute(cv.UMat(grayL_down),cv.UMat(grayR_down))
dispr = right_matcher.compute(cv.UMat(grayR_down),cv.UMat(grayL_down))
displ = np.int16(cv.UMat.get(displ))
dispr = np.int16(cv.UMat.get(dispr))
# --------------------------------------------------------------------------

# WLS FILTER ----------------------------------------------------------------
filtered_disp = wls_filter.filter(displ, rectifiedL, None, dispr)
# --------------------------------------------------------------------------

# # MASK ---------------------------------------------------------------------
# # CONFIDENCE MAP
conf_map=wls_filter.getConfidenceMap()
# # ROI
ROI = np.array(wls_filter.getROI())*downscale
# #MASK
mask=np.zeros(conf_map.shape, conf_map.dtype)
mask[ROI[1]:ROI[1]+ROI[3], ROI[0]:ROI[0]+ROI[2]]=1
# # --------------------------------------------------------------------------

#
# # # BILATERAL FILTER ---------------------------------------------------------
fbs_spatial=30.0
fbs_luma=8.0
fbs_chroma=8.0
fbs_lambda=128.0
solved_filtered_disp = cv.ximgproc.fastBilateralSolverFilter(rectifiedL, filtered_disp, conf_map/255.0, None, fbs_spatial, fbs_luma, fbs_chroma, fbs_lambda)
# # # --------------------------------------------------------------------------

# REPROJECT TO 3D ----------------------------------------------------------
# format disparity
disparity=solved_filtered_disp.astype(np.float32)/16.0*mask

# Note: If one uses Q obtained by stereoRectify, then the returned points are represented in the first camera's rectified coordinate system
cloud = cv.reprojectImageTo3D(disparity, Q, handleMissingValues=True)

# depth map
depth_map=cloud[:,:,2]*mask

plt.figure()
plt.imshow(disparity)
plt.title('carte de disparité')
plt.colorbar()
plt.savefig('carte_d_zz.png')

plt.figure()
plt.imshow(depth_map)
plt.title('carte de profondeur')
plt.colorbar()
plt.savefig('carte_p_zz.png')
# -------------------------------------------------------------------------



# SAVEGARDER MESH ----------------------------------------------------------
# colors = cv.cvtColor(rectifiedL, cv.COLOR_BGR2RGB)
# colors_valides = colors[mask.astype(bool)]
# points_valides=cloud[mask.astype(bool)]
# mask2 = (points_valides[:,2]<2).astype(bool)
# out_fn = '3dpoints/{}.ply'.format('dents')
# write_ply(out_fn, points_valides[mask2], colors_valides[mask2])
# --------------------------------------------------------------------------
