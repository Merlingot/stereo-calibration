import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import glob, os

from util import readXML, find_corners, read_images, write_ply


# Fichiers de calibration:
left_xml='cam1.xml'
right_xml='cam2.xml'
# Images à reconstruire:
i='38' #numero de l'image
left="stereo/left{}.jpg".format(i)
right="stereo/right{}.jpg".format(i)

# def return_point3d(left, right, left_xml, right_xml, wls_lambda=8000.0, wls_sigma=1.5, i=0):

""" Calculer les points 3d d'une image  -> à visualiser dans MeshLab

Arguments:
    left (str): nom du fichier de l'image de gauche
    right (str) : nom du fichier de l'image de droite
    left_xml (str): nom du fichier de calibration de l'image de gauche
    right_xml (str) : ...
    wls_lambda (float) : valeur du paramètre lambda pour le filtre wls. Le range [1000,8000] fonctionne
    wls_sigma (float) : valeur du paramètre sigma pour le filtre wls
    Le range [1.5,5] fonctionne bien -> Pour des images avec grosse résolution >1000x1000 augmenter sigma_wls à 5
    i (int) : numero des images analysée dans une série d'image. Les points 3d sont enregistrés sous "./3dpoints/out_{i}.ply"
"""

# Lire les fichiers de calibration
K1,D1,_,_,imageSize1, E, F=readXML(left_xml) #left
K2,D2,R, T,imageSize2, _, _=readXML(right_xml) #right
# On prend la taille de la caméra de gauche (référence)
width, height=imageSize1[1],imageSize1[0]

# Lire les images
colorL, grayL=read_images(left) #left
colorR, grayR=read_images(right) #right

R1, R2, P1, P2, Q , roi_1, roi_2= cv.stereoRectify(K1, D1, K2, D2, (height,width), R, T, flags=0)
leftMapX, leftMapY = cv.initUndistortRectifyMap(K1, D1, R1, P1, (width, height), cv.CV_32FC1)
left_rectified = cv.remap(grayL, leftMapX, leftMapY, cv.INTER_LINEAR)
color_rectified=cv.remap(colorL, leftMapX, leftMapY, cv.INTER_LINEAR)
rightMapX, rightMapY = cv.initUndistortRectifyMap(K2, D2, R2, P2, (width, height), cv.CV_32FC1)
right_rectified = cv.remap(grayR, rightMapX, rightMapY, cv.INTER_LINEAR)

# Stereo mathing ---------------------------------------------
plt.imshow(left_rectified)
plt.imshow(right_rectified)

left_for_matcher = left_rectified
right_for_matcher = right_rectified
d=np.absolute((P1-P2)[0][2])
wsize = 3 #sgbm
if d :
    min_disp=0
    num_disp=int(122 - min_disp)
# else:
#     min_disp=int(d + (16 - d%16))
#     num_disp=int(122 - min_disp)

left_matcher = cv.StereoSGBM_create(min_disp, num_disp, wsize);
left_matcher.setP1(24*wsize*wsize);
left_matcher.setP2(96*wsize*wsize);
left_matcher.setPreFilterCap(0);
left_matcher.setMode(cv.StereoSGBM_MODE_HH);
left_matcher.setSpeckleRange(0)
left_matcher.setSpeckleWindowSize(0)
right_matcher = cv.ximgproc.createRightMatcher(left_matcher);

left_disp=left_matcher.compute(left_for_matcher, right_for_matcher).astype(np.float32) / 16.0
right_disp=right_matcher.compute(right_for_matcher, left_for_matcher).astype(np.float32) / 16.0

plt.imshow(left_disp)

# Filter -----------------------------------------------------

# wls_filter -----
wls_lambda=8000.0
wls_sigma=1.5 #5
wls_filter = cv.ximgproc.createDisparityWLSFilter(left_matcher)
wls_filter.setLambda(wls_lambda);
wls_filter.setSigmaColor(wls_sigma);
filtered_disp = wls_filter.filter(left_disp, colorL, None, right_disp, None, colorR);
conf_map = wls_filter.getConfidenceMap()

plt.figure()
plt.imshow(filtered_disp)
plt.colorbar()

# fbs_filter ------

fbs_spatial=8.0
fbs_luma=8.0
fbs_chroma=8.0
fbs_lambda=128.0
solved_filtered_disp = cv.ximgproc.fastBilateralSolverFilter(color_rectified, filtered_disp, conf_map/255.0, None, fbs_spatial, fbs_luma, fbs_chroma, fbs_lambda)

plt.figure()
plt.imshow(solved_filtered_disp)
plt.colorbar()

# assert np.linalg.norm(solved_filtered_disp)!=np.NaN, 'Échec filtre. Changer paramètre filtre wls'
# cv.imwrite('disparity_map.png'.format(i), solved_filtered_disp)

# Reproject to 3d --------------------------------------------
points = cv.reprojectImageTo3D(filtered_disp, Q, handleMissingValues=False)

plt.figure()
plt.imshow(points[:,:,2])
plt.colorbar()

# Save ----------------------------------------------------
colors = cv.cvtColor(color_rectified, cv.COLOR_BGR2RGB)
mask = solved_filtered_disp > solved_filtered_disp.min()
# mask = solved_filtered_disp < solved_filtered_disp.max()

out_points = points[mask]
out_colors = colors[mask]
out_fn = '3dpoints/out_{}.ply'.format(i)
write_ply(out_fn, out_points, out_colors)


# return_point3d(left, right, left_xml, right_xml, wls_lambda=8000.0, wls_sigma=1.5, i=i)
