import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import glob, os

from util import readXML, find_corners, read_images, write_ply, refine_corners

# Fichiers de calibration:
left_xml='cam1.xml'
right_xml='cam2.xml'
# Images à reconstruire:
i='30' #numero de l'image
left="stereo/left{}.jpg".format(i)
right="stereo/right{}.jpg".format(i)
# Damier
patternSize=(9,6)
squaresize=3.64e-2

# LIRE FICHIERS DE CALIBRATION -------------------------------------------------
K1,D1,_,_,imageSize1, E, F=readXML(left_xml) #left
K2,D2,R, T,imageSize2, _, _=readXML(right_xml) #right
# La taille de la caméra de gauche est celle de référence
width1, height1=imageSize1[1],imageSize1[0]
width2, height2=imageSize2[1],imageSize2[0]
# ------------------------------------------------------------------------------

# RECTIFICATION ----------------------------------------------------------------
R1, R2, P1, P2, Q , _, _= cv.stereoRectify(K1, D1, K2, D2, (height1, width1), R, T, flags=0, alpha=0)
# map_left_x, map_left_y = cv.initUndistortRectifyMap(K1, D1, R1, P1, (width1, height1), cv.CV_32FC1)
# map_right_x, map_right_y = cv.initUndistortRectifyMap(K2, D2, R2, P2, (width2, height2), cv.CV_32FC1)
# left_frame_rect = cv2.remap(left_frame, map_left_x, map_left_y, interpolation=cv2.INTER_LINEAR)
# right_frame_rect = cv2.remap(right_frame, map_right_x, map_right_y, interpolation=cv2.INTER_LINEAR)
# ------------------------------------------------------------------------------

# LIRE IMAGES ET TROUVER POINTS ------------------------------------------------
_, grayl=read_images(left)
_, grayr=read_images(right)
# damier
objp = np.zeros((patternSize[0]*patternSize[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:patternSize[0], 0:patternSize[1]].T.reshape(-1, 2)
objp*=squaresize
# trouver
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
ret_l, corners_l = cv.findChessboardCorners(grayl, patternSize, None)
ret_r, corners_r = cv.findChessboardCorners(grayr, patternSize, None)
if ret_l*ret_r :
    pts_l= cv.cornerSubPix(grayl, corners_l, (11, 11),(-1, -1), criteria)
    pts_r= cv.cornerSubPix(grayr, corners_r, (11, 11),(-1, -1), criteria)
# ------------------------------------------------------------------------------


# MATRICES DE PROJECTION -------------------------------------------------------
ret, rvec1, t1 = cv.solvePnP(objp, pts_l, K1, D1 )
ret, rvec2, t2 = cv.solvePnP(objp, pts_r, K2, D2 )
r1=cv.Rodrigues(rvec1)[0]; r2=cv.Rodrigues(rvec2)[0]
p1=np.column_stack((r1,t1)); Proj1=K1@p1
p2=np.column_stack((r2,t2)); Proj2=K2@p2
# ------------------------------------------------------------------------------

# UNDISTORT POINTS -------------------------------------------------------------
ptsl=cv.undistortPoints(pts_l, K1, D1, None, None, K1) #not rectification
ptsr=cv.undistortPoints(pts_r, K2, D2, None, None, K2) #not rectification
N=ptsl.shape[0]
projPoints1 = np.zeros((2,N))
projPoints2 = np.zeros((2,N))
for i in range(N):
    projPoints1[:,i]=ptsl[i,0]
    projPoints2[:,i]=ptsr[i,0]
# ------------------------------------------------------------------------------

# TRIANGULATION ----------------------------------------------------------------
#triangulatePoints: If the projection matrices from stereoRectify are used, then the returned points are represented in the first camera's rectified coordinate system.
points4D=cv.triangulatePoints(Proj1, Proj2, projPoints1, projPoints2)
points3D=cv.convertPointsFromHomogeneous(points4D.T)

X,Y,Z=points3D[:,0,0], points3D[:,0,1], points3D[:,0,2]
# ------------------------------------------------------------------------------
#
# plt.plot(pts_l[:,0,0], pts_l[:,0,1], '.-')
# plt.plot(ptsl[:,0,0], ptsl[:,0,1], '.')


# CALCUL DES ERREURS -----------------------------------------------------------
# plt.figure()
# plt.plot(objp[:,0],objp[:,1], 'bo-' )
# plt.plot(X,Y, 'r.')

patternSizeT=(patternSize[1],patternSize[0])
x,y,z=np.zeros((patternSizeT)),np.zeros((patternSizeT)),np.zeros((patternSizeT))
xo,yo,zo=np.zeros((patternSizeT)),np.zeros((patternSizeT)),np.zeros((patternSizeT))
j=0
for i in range(patternSize[1]):
    x[i,:]=X[j:j+9]
    y[i,:]=Y[j:j+9]
    z[i,:]=Z[j:j+9]
    xo[i,:]=objp[j:j+9,0]
    yo[i,:]=objp[j:j+9,1]
    zo[i,:]=objp[j:j+9,2]
    j+=9

errX=(xo-x)
errY=(yo-y)
errZ=(zo-z)
#
plt.figure()
plt.imshow(errX*1e3)
plt.colorbar()
#
plt.figure()
plt.imshow(errY*1e3)
plt.colorbar()
#
plt.figure()
plt.imshow(errZ*1e3)
plt.colorbar()

errXrms=np.sqrt((errX**2).mean())*1e3
errYrms=np.sqrt((errY**2).mean())*1e3
errZrms=np.sqrt((errZ**2).mean())*1e3
errRMS=np.sqrt( (errX**2 + errY**2 + errZ**2).mean() )
