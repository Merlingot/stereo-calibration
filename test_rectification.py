from modules.util import *
from modules.points3d import *
import cv2 as cv

from config import calibration_folder, left_xml, right_xml, patternSize

################################################################################
# Choisir une image spécifique à analyser:
left='{}left002.jpg'.format(calibration_folder)
right='{}right002.jpg'.format(calibration_folder)
################################################################################



_,_, _, _ ,_,E, F = readXML(left_xml) # left

cam1,cam2=get_cameras(left_xml, right_xml, alpha=-1)
cam1.set_images(left)
cam2.set_images(right)

not_rectifiedL=cv.cvtColor(cam1.not_rectified, cv.COLOR_RGB2GRAY)
not_rectifiedR=cv.cvtColor(cam2.not_rectified, cv.COLOR_RGB2GRAY)

img1=not_rectifiedL; img2=not_rectifiedR
ret, pts1=cv.findChessboardCorners(not_rectifiedL, patternSize, None)
ret, pts2 = cv.findChessboardCorners(not_rectifiedR, patternSize, None)
assert len(pts1)>0 and len(pts2)>0, 'coins non détectés, choisir autre image'

N=patternSize[0]*patternSize[1]
pts1=pts1[[0,int(N/2), int(N-1)],:,:]; pts2=pts2[[0,int(N/2),int(N-1)],:,:]
# Find epilines corresponding to points in right image (second image) and
# drawing its lines on left image
lines1 = cv.computeCorrespondEpilines(pts2, 2,F)
lines1 = lines1.reshape(-1,3)
img5,img6 = drawlines(img1,img2,lines1,pts1,pts2)
# Find epilines corresponding to points in left image (first image) and
# drawing its lines on right image
lines2 = cv.computeCorrespondEpilines(pts1, 1,F)
lines2 = lines2.reshape(-1,3)
img3,img4 = drawlines(img2,img1,lines2,pts2,pts1)


plt.figure(figsize=(10,5))
plt.title('Avant rectification')
plt.imshow(np.concatenate((img5,img3), axis=1))
plt.show()

rectifiedL = cv.remap(img5, cam1.map_x, cam1.map_y, interpolation=cv.INTER_LINEAR)
rectifiedR = cv.remap(img3, cam2.map_x, cam2.map_y, interpolation=cv.INTER_LINEAR)

plt.figure(figsize=(10,5))
plt.title('Après rectification')
plt.imshow(np.concatenate((rectifiedL,rectifiedR), axis=1))
plt.show()
