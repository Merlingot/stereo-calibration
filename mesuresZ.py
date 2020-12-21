import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import glob, os

from util import readXML, find_corners, read_images, write_ply, refine_corners

# Fichiers de calibration:
left_xml='cam1.xml'
right_xml='cam2.xml'
# Images à reconstruire:
i='38' #numero de l'image
left="stereo/left{}.jpg".format(i)
right="stereo/right{}.jpg".format(i)
# Damier
patternSize=(9,6)
squaresize=3.64e-2

# Lire les fichiers de calibration
K1,D1,_,_,imageSize1, E, F=readXML(left_xml) #left
K2,D2,R, T,imageSize2, _, _=readXML(right_xml) #right
# Calcul des matrices de projection
P1 = np.array([ [1,0,0,0],[0,1,0,0],[0,0,1,0] ])
P2 = np.array([ [R[0][0],R[0][1],R[0][2],T.item(0)],[R[1][0],R[1][1],R[1][2],T.item(1)],[R[2][0],R[2][1],R[2][2],T.item(2) ]])

# Trouver les points du damier
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

plt.plot(pts_l[:,0,0], pts_l[:,0,1], 'o')



# # undistort the points
# ptsl=cv.undistortPoints(pts_l, K1,D1)
# ptsr=cv.undistortPoints(pts_r, K2,D2)
#
# plt.plot(ptsl[:,0,0], ptsl[:,0,1], 'o')

# trianguler les points
points4D=cv.triangulatePoints(P1, P2, pts_l, pts_r)
src = points4D.reshape(54,4)
points3D=cv.convertPointsFromHomogeneous(src)
X,Y,Z = points3D[:,0,0], points3D[:,0,1], points3D[:,0,2]


plt.figure()
plt.plot(X,Y, 'o')
