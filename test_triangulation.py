from modules.util import *
from modules.points3d import *


# Choisir une image à analyser -------------------------------------------------
fleft = 'captures/captures_calibration/left27.jpg'
fright = 'captures/captures_calibration/right27.jpg'
# ------------------------------------------------------------------------------

# Fichiers de calibration ------------------------------------------------------
left_xml='cam1.xml'
right_xml='cam2.xml'
# Damier -----------------------------------------------------------------------
patternSize=(10,8)
squaresize=2e-2
# ------------------------------------------------------------------------------


cam1,cam2=get_cameras(left_xml, right_xml)
cam1.set_images(fleft)
cam2.set_images(fright)

# CALCUL DE TOUS LES COINS THÉORIQUES POUR LA CAMÉRA 1 -------------------------

# Référentiel monde
objp=coins_damier(patternSize,squaresize)
world=objp.T

# Référentiel caméra non rectifié
ret, r ,t =find_rt(patternSize, objp, cam1.not_rectified, cam1.K, cam1.D)
unrec1=r@world+t #coins théoriques dans ref cam1 non rectifiée

# # Référentiel image non rectifié sans distortion
# corners_unrec1, _ = cv.projectPoints(unrec1, np.zeros((3,1)), np.zeros((3,1)), cam1.K, cam1.D) #points théoriques dans ref image cam rectifiée

# Référentiel rectifié
rec1=cam1.R@unrec1 #points théoriques dans ref cam rectifiée

# Référentiel image rectifié avec distortion ?
# corners_rec1, _ = cv.projectPoints(rec1, np.zeros((3,1)), np.zeros((3,1)), cam1.P[:,:3], np.zeros((1,4))) #points théoriques dans ref image cam rectifiée
# ------------------------------------------------------------------------------


# POINTS IMAGES DÉTECTÉS (LA MESURE) -------------------------------------------
# Référentiel rectifié
grayl=cv.cvtColor(cam1.rectified, cv.COLOR_BGR2GRAY)
grayr=cv.cvtColor(cam2.rectified, cv.COLOR_BGR2GRAY)
# Détection des coins
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
ret_l, corners_l = cv.findChessboardCorners(grayl, patternSize, None)
ret_r, corners_r = cv.findChessboardCorners(grayr, patternSize, None)
if ret_l*ret_r :
    corners_rec_l= cv.cornerSubPix(grayl, corners_l, (11, 11),(-1, -1), criteria)
    corners_rec_r= cv.cornerSubPix(grayr, corners_r, (11, 11),(-1, -1), criteria)
# ------------------------------------------------------------------------------

# TRIANGULATION -----------------------------------------------------------
# Formatage
N=corners_rec_l.shape[0]
projPoints1 = np.zeros((2,N));
projPoints2 = np.zeros((2,N))
for i in range(N):
    projPoints1[:,i]=corners_rec_l[i,0];
    projPoints2[:,i]=corners_rec_r[i,0]


# triangulatePoints: If the projection matrices from stereoRectify are used, then the returned points are represented in the first camera's rectified coordinate system.
points4D=cv.triangulatePoints(cam1.P, cam2.P, projPoints1, projPoints2)
points3D=cv.convertPointsFromHomogeneous(points4D.T)
X,Y,Z=points3D[:,0,0], points3D[:,0,1], points3D[:,0,2]
points_rec1=np.stack((X,Y,Z)) # Référentiel rectifié
points_unrec1 = cam1.R.T@points_rec1 # Référentiel non rectifié
points_monde1 = r.T@(points_unrec1-t) # Référentiel monde
# ------------------------------------------------------------------------------

# GRAPHIQUES -------------------------------------------------------------------

# RÉFÉRENTIEL REC
plt.figure()
plt.title('Triangulation plan x-y')
plt.xlabel('x')
plt.ylabel('y')
plt.plot(rec1[0,:], rec1[1,:], 'o-') # théorique
plt.plot(points_rec1[0,:], points_rec1[1,:], '.-') # détecté
plt.savefig('xy27.png')

plt.figure()
plt.title('Triangulation plan x-z')
plt.xlabel('x')
plt.ylabel('z')
plt.plot(rec1[0,:], rec1[2,:], 'o-')
plt.plot(points_rec1[0,:], points_rec1[2,:], '.-')
plt.savefig('xz27.png')

plt.figure()
plt.title('Triangulation plan y-z')
plt.xlabel('y')
plt.ylabel('z')
plt.plot(rec1[1,:], rec1[2,:], 'o-')
plt.plot(points_rec1[1,:], points_rec1[2,:], '.-')
plt.savefig('yz27.png')

# RÉFÉRENTIEL MONDE
# plt.plot(world[0,:], world[1,:], 'o-') # théorique
# plt.plot(points_monde1[0,:], points_monde1[1,:], '.-') # détecté
#
# plt.plot(world[0,:], world[2,:], 'o-')
# plt.plot(points_monde1[0,:], points_monde1[2,:], '.-')
#
# plt.plot(world[1,:], world[2,:], 'o-')
# plt.plot(points_monde1[1,:], points_monde1[2,:], '.-')
# ------------------------------------------------------------------------------
