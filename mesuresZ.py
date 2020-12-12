import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import glob, os

from util import readXML, find_corners, read_images, write_ply, refine_corners



# Fichiers de calibration:
left_xml='cam1.xml'
right_xml='cam2.xml'
# Images à reconstruire:
i='15' #numero de l'image
left="stereo/left{}.jpg".format(i)
right="stereo/right{}.jpg".format(i)
# Damier
patternSize=(9,6)
squaresize=3.64e-2



# Lire les fichiers de calibration
K1,D1,_,_,imageSize1, E, F=readXML(left_xml) #left
K2,D2,R, T,imageSize2, _, _=readXML(right_xml) #right
# On prend la taille de la caméra de gauche (référence)
width, height=imageSize1[1],imageSize1[0]
R1, R2, P1, P2, Q , roi_1, roi_2= cv.stereoRectify(K1, D1, K2, D2, (height,width), R, T, flags=0, alpha=-1)


# damier
objp = np.zeros((patternSize[0]*patternSize[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:patternSize[0], 0:patternSize[1]].T.reshape(-1, 2)
objp*=squaresize
# criteria:
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# liste de points
ret_l, corners_l, color_l= find_corners(left,patternSize)
ret_r, corners_r, color_r = find_corners(right,patternSize)
if ret_l*ret_r :
    ptsl= cv.cornerSubPix(cv.cvtColor(color_l, cv.COLOR_BGR2GRAY), corners_l, (11, 11),(-1, -1), criteria)
    ptsr= cv.cornerSubPix(cv.cvtColor(color_r, cv.COLOR_BGR2GRAY), corners_r, (11, 11),(-1, -1), criteria)
    _ = cv.drawChessboardCorners(color_l, patternSize, ptsl, True)
    fname='left.jpg'
    cv.imwrite(fname, color_l)
    _ = cv.drawChessboardCorners(color_r, patternSize, ptsr, True)
    fname='right.jpg'
    cv.imwrite(fname, color_r)


points4D=cv.triangulatePoints(P1, P2, ptsl, ptsr)
src = points4D.reshape(54,4)
points3D=cv.convertPointsFromHomogeneous(src)
X,Y,Z = points3D[:,0,0], points3D[:,0,1], points3D[:,0,2]


from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X, Y, Z)
plt.show()
