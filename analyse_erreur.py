
from modules.points3d import *
from modules.util import *

# Fichiers de calibration ------------------------------------------------------
left_xml='cam1.xml'
right_xml='cam2.xml'
patternSize=(10,8)
squaresize=2e-2
# ------------------------------------------------------------------------------


# Déclaration des listes d'erreur ----------------------------------------------
N=[]
errtot1 = []
errtot2 = []
errxyz1 = []
errxyz2 = []
errcyl1 = []
errcyl2 = []
# Damier et caméras ------------------------------------------------------------
objp = coins_damier(patternSize,squaresize)
world = objp.T
cam1, cam2 = get_cameras(left_xml, right_xml)
# ------------------------------------------------------------------------------


for nb in range(1,10):

    N.append(nb)
    # Images à reconstruire:
    left="captures/captures_calibration/left{}.jpg".format(nb)
    right="captures/captures_calibration/right{}.jpg".format(nb)
    cam1.set_images(left)
    cam2.set_images(right)

    # CALCUL AVEC CARTE DE DISPARITÉ  ------------------------------------------
    cloud, mask, depth_map= calcul_mesh(cam1.rectified, cam2.rectified, cam1.Q)
    save_mesh(cam1.rectified, cloud, mask, 'mesh_{}'.format(nb))
    pts_rec, corners_rec = coins_mesh(patternSize, cam1.rectified, cloud) # Points détectés

    ret, r, t = find_rt(patternSize, objp, cam1.not_rectified, cam1.K, cam1.D)
    rec = get_rec(objp, r, t, cam1.R, cam1.P ) # Points théoriques

    errtot, errxyz, errcyl = err_points(patternSize, rec, pts_rec)
    errtot1.append(errtot)
    errxyz1.append(errxyz)
    errcyl1.append(errcyl)
    # --------------------------------------------------------------------------

    # CALCUL AVEC TRIANGULATION  ----------------------------------------------
    re = triangulation_rec(patternSize, cam1.rectified, cam2.rectified, cam1.P, cam2.P ) # Points détectés
    errtot, errxyz, errcyl = err_points(patternSize, rec, re)
    errtot2.append(errtot)
    errxyz2.append(errxyz)
    errcyl2.append(errcyl)
    # --------------------------------------------------------------------------


plt.figure()
plt.title('Erreur rms sur le rayon')
plt.xlabel('# image')
plt.ylabel('Erreur (mm)')
plt.ylim(0,25)
plt.plot(N, np.array(errcyl1)[:,0]*1e3, 'bo-')
plt.plot( N, np.array(errcyl2)[:,0]*1e3, 'ro-')
plt.legend(['carte de disparité', 'triangulation'])

plt.figure()
plt.title('Erreur rms sur la position en z')
plt.xlabel('# image')
plt.ylabel('Erreur (mm)')
plt.ylim(0,50)
plt.plot(N, np.array(errxyz1)[:,2]*1e3, 'bo-')
plt.plot( N, np.array(errxyz2)[:,2]*1e3, 'ro-')
plt.legend(['carte de disparité', 'triangulation'])
