from util import *
from points3d import *

# Fichiers de calibration:
left_xml='cam1.xml'
right_xml='cam2.xml'
cam1, cam2 = get_cameras(left_xml, right_xml)

patternSize=(10,8)
squaresize=2e-2
folder='captures_flip'

world_th=coins_damier(patternSize,squaresize).T

# RESOLUTION ET TAILLE -----------------------------------------------------
resolution='VGA'
width,height = get_image_dimension_from_resolution(resolution)
image_size = Resolution(width,height)
# --------------------------------------------------------------------------

# LIRE FICHIERS DE CALIBRATION ---------------------------------------------
K1,d1, _, _ ,_,E, F = readXML(left_xml) # left
K2,d2, R, T ,_, _, _ = readXML(right_xml) # right
# --------------------------------------------------------------------------

# RECTIFICATION ------------------------------------------------------------
R1, R2, P1, P2, Q, _, _ = cv.stereoRectify(cameraMatrix1=K1, cameraMatrix2=K2,distCoeffs1=d1,distCoeffs2=d2,R=R, T=T, flags=0, alpha=1,
imageSize=(image_size.height, image_size.width))

map_left_x, map_left_y = cv.initUndistortRectifyMap(K1, d1, R1, P1, (image_size.width, image_size.height), cv.CV_32FC1)
map_right_x, map_right_y = cv.initUndistortRectifyMap(K2, d2, R2, P2, (image_size.width, image_size.height), cv.CV_32FC1)
# --------------------------------------------------------------------------

# CAMÉRAS ------------------------------------------------------------------
cam1 = Camera(K1,d1,R1,P1,map_left_x,map_left_y); cam1.Q=Q
cam2 = Camera(K2,d2,R2,P2,map_right_x,map_right_y)
cam1.set_images('{}/captures_erreur/left4.jpg'.format(folder))
cam2.set_images('{}/captures_erreur/right4.jpg'.format(folder))
# --------------------------------------------------------------------------
not_rectifiedL=cv.cvtColor(cam1.not_rectified, cv.COLOR_RGB2GRAY)
not_rectifiedR=cv.cvtColor(cam2.not_rectified, cv.COLOR_RGB2GRAY)


img1=not_rectifiedL; img2=not_rectifiedR
ret, pts1=cv.findChessboardCorners(not_rectifiedL, patternSize, None)
ret, pts2 = cv.findChessboardCorners(not_rectifiedR, patternSize, None)
assert len(pts1)>0 and len(pts2)>0, 'coins non détectés, choisir autre image'

pts1=pts1[[1,20,30],:,:]; pts2=pts2[[1,20,30],:,:]
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
