from modules.util import *
from modules.points3d import *
import matplotlib.pyplot as plt

################################################################################
################################################################################
# Choisir une image à analyser -------------------------------------------------
left='data/12mm/cibles/left.jpg'
right='data/12mm/cibles/right.jpg'
# ------------------------------------------------------------------------------
# Fichiers de calibration ------------------------------------------------------
left_xml='data/12mm/cam1_cibles.xml'
right_xml='data/12mm/cam2_cibles.xml'

# Coordonées des cibles
fname_cibles="data/12mm/cibles/objpts.txt"
fname_cibles_l="data/12mm/cibles/pts_left.txt"
fname_cibles_r="data/12mm/cibles/pts_right.txt"
################################################################################
patternSize=(15,10)
squaresize=7e-2
################################################################################

# Caméras
cam1, cam2 = get_cameras(left_xml, right_xml, alpha=-1)
cam1.set_images(left)
cam2.set_images(right)

# CALCUL DE TOUS LES COINS THÉORIQUES POUR LA CAMÉRA 1 -------------------------
# Référentiel image non rectifié
cibles_l=np.genfromtxt(fname_cibles_l).astype(np.float32)
cibles_l=cibles_l.reshape(cibles_l.shape[0], 1, 2)
cibles_r=np.genfromtxt(fname_cibles_r).astype(np.float32)
cibles_r=cibles_r.reshape(cibles_r.shape[0], 1, 2)
# Référentiel monde
cibles = np.genfromtxt(fname_cibles).astype(np.float32)
cibles=cibles[:cibles_l.shape[0], :]
world=cibles.T
# # Référentiel image rectifié sans distortion
corners_rec_l=cv.undistortPoints(cibles_l, cam1.K, cam1.D, None, cam1.R, cam1.P)
corners_rec_r=cv.undistortPoints(cibles_r, cam2.K, cam2.D, None, cam2.R, cam2.P)

# Référentiel caméra non rectifié
ret, rvec, t = cv.solvePnP(cibles, cibles_l, cam1.K, cam1.D )
r,_=cv.Rodrigues(rvec)
unrec1=r@world+t #coins théoriques dans ref cam1 non rectifiée
unrec2=cam2.r12@unrec1+cam2.t12

# Référentiel rectifié
rec1=cam1.R@unrec1 #points théoriques dans ref cam rectifiée
rec2=cam2.R@unrec2 #points théoriques dans ref cam rectifiée

# # Référentiel image rectifié avec distortion ?
corners_rec1, _ = cv.projectPoints(rec1, np.zeros((3,1)), np.zeros((3,1)), cam1.P[:,:3], cam1.D)  #points théoriques dans ref image cam rectifiée
corners_rec2, _ = cv.projectPoints(rec2, np.zeros((3,1)), np.zeros((3,1)), cam2.P[:,:3],cam2.D )  #points théoriques dans ref image cam rectifiée
# ------------------------------------------------------------------------------

# plt.imshow(cam1.rectified)
# plt.plot(corners_rec_l[:,0,0],corners_rec_l[:,0,1])
# plt.plot(corners_rec1[:,0,0],corners_rec1[:,0,1])
#
# plt.imshow(cam2.rectified)
# plt.plot(corners_rec_r[:,0,0],corners_rec_r[:,0,1])
# plt.plot(corners_rec2[:,0,0],corners_rec2[:,0,1])


# TRIANGULATION EXPÉRIMENTALE --------------------------------------------------
# Formatage
N=corners_rec_l.shape[0]
projPoints1 = np.zeros((2,N));
projPoints2 = np.zeros((2,N))
for i in range(N):
    projPoints1[:,i]=corners_rec_l[i,0]
    projPoints2[:,i]=corners_rec_r[i,0]
points4D=cv.triangulatePoints(cam1.P, cam2.P, projPoints1, projPoints2)
points3D=cv.convertPointsFromHomogeneous(points4D.T)
X,Y,Z=points3D[:,0,0], points3D[:,0,1], points3D[:,0,2]
points_rec_l=np.stack((X,Y,Z)) # Référentiel rectifié
# ------------------------------------------------------------------------------

# TRIANGULATION THÉORIQUE ------------------------------------------------------
# Formatage
N=corners_rec1.shape[0]
projPoints1 = np.zeros((2,N));
projPoints2 = np.zeros((2,N))
for i in range(N):
    projPoints1[:,i]=corners_rec1[i,0]
    projPoints2[:,i]=corners_rec2[i,0]
points4D=cv.triangulatePoints(cam1.P, cam2.P, projPoints1, projPoints2)
points3D=cv.convertPointsFromHomogeneous(points4D.T)
X,Y,Z=points3D[:,0,0], points3D[:,0,1], points3D[:,0,2]
points_rec1=np.stack((X,Y,Z)) # Référentiel rectifié
# ------------------------------------------------------------------------------

# GRAPHIQUES -------------------------------------------------------------------
# RÉFÉRENTIEL REC
plt.figure()
plt.title('Triangulation plan x-y')
plt.xlabel('x')
plt.ylabel('y')
plt.plot(rec1[0,:], rec1[1,:], 'o-')  # théorique
plt.plot(points_rec_l[0,:], points_rec_l[1,:], '.-')  # détecté
# plt.plot(points_rec1[0,:], points_rec1[1,:], '.-')  # théorique
# plt.savefig('figures/xy.png')

plt.figure()
plt.title('Triangulation plan x-z')
plt.xlabel('x')
plt.ylabel('z')
plt.plot(rec1[0,:], rec1[2,:], 'o-')
plt.plot(points_rec_l[0,:], points_rec_l[2,:], '.-')
# plt.plot(points_rec1[0,:], points_rec1[2,:], '.-')
# plt.savefig('figures/xz.png')

plt.figure()
plt.title('Triangulation plan y-z')
plt.xlabel('y')
plt.ylabel('z')
plt.plot(rec1[1,:], rec1[2,:], 'o-')
plt.plot(points_rec_l[1,:], points_rec_l[2,:], '.-')
# plt.plot(points_rec1[1,:], points_rec1[2,:], '.-')
# plt.savefig('figures/yz.png')

# ------------------------------------------------------------------------------
