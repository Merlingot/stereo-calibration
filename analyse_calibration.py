
from modules.points3d import *
from modules.util import *

################################################################################
# Damier -----------------------------------------------------------------------
patternSize=(15,10)
squaresize=7e-2
objp = coins_damier(patternSize,squaresize)
world = objp.T
# Images -----------------------------------------------------------------------
left=np.concatenate( (np.sort(glob.glob("captures_zed/captures_3/left*.jpg")) , np.sort(glob.glob("captures_zed/captures_2/left*.jpg"))[0:20]))
left
right=np.concatenate( (np.sort(glob.glob("captures_zed/captures_3/right*.jpg")) , np.sort(glob.glob("captures_zed/captures_2/right*.jpg"))[0:20]))
################################################################################

# Déclaration des listes d'erreur ----------------------------------------------
N=[]
errtot1 = []; errtot2 = []
Z=[];z1=[];z2=[]
X=[];x1=[];x2=[]
Y=[];y1=[];y2=[]
# Caméras ----------------------------------------------------------------------
left_xml='cam1_zed.xml'
right_xml='cam2_zed.xml'
cam1, cam2 = get_cameras(left_xml, right_xml, alpha=0)
# ------------------------------------------------------------------------------

for nb in range(len(left)):

    # Images à reconstruire:
    cam1.set_images(left[nb])
    cam2.set_images(right[nb])
    # CALCUL AVEC CARTE DE DISPARITÉ  ------------------------------------------
    cloud, mask, depth_map= calcul_mesh(cam1.rectified, cam2.rectified, cam1.Q)
    # save_mesh(cam1.rectified, cloud, mask, 'mesh_{}'.format(nb))
    pts_rec, corners_rec = coins_mesh(patternSize, cam1.rectified, cloud, winSize=(3,3)) # Points détectés

    if pts_rec is not None:
        N.append(nb)
        ret, r, t = find_rt(patternSize, objp, cam1.not_rectified, cam1.K, cam1.D, winSize=(3,3))
        rec = get_rec(objp, r, t, cam1.R, cam1.P ) # Points théoriques

        errtot, errxyz, errcyl, vecerr = err_points(patternSize, rec, pts_rec)
        errtot1.append(errtot)
        x0, x, y0, y, z0, z = vecerr
        X.append((x0))
        x1.append((x))
        Y.append((y0))
        y1.append((y))
        Z.append((z0))
        z1.append((z))
    # --------------------------------------------------------------------------

errz = []
Zmean=[]
zmean=[]
for i in range(len(Z)):
    zi = np.mean(z1[i])
    Zi = np.mean(Z[i])
    ei = np.absolute(Zi-zi)
    errz.append(ei)
    Zmean.append(Zi)
    zmean.append(zi)
Zmean=np.array(Zmean)
errz=np.array(errz)
a=np.argsort(Zmean)

################################################################################
# Caméras ----------------------------------------------------------------------
left_xml='cam1.xml'
right_xml='cam2.xml'
cam1, cam2 = get_cameras(left_xml, right_xml, alpha=0)
# ------------------------------------------------------------------------------
# Déclaration des listes d'erreur ----------------------------------------------
errtot1 = []; errtot2 = []
Z=[];z1=[];z2=[]
X=[];x1=[];x2=[]
Y=[];y1=[];y2=[]
NN=[]
for nb in range(len(left)):
    # Images à reconstruire:
    cam1.set_images(left[nb])
    cam2.set_images(right[nb])

    # CALCUL AVEC CARTE DE DISPARITÉ  ------------------------------------------
    NN.append(nb)
    cloud, mask, depth_map= calcul_mesh(cam1.rectified, cam2.rectified, cam1.Q)
    # save_mesh(cam1.rectified, cloud, mask, 'mesh_{}'.format(nb))
    pts_rec, corners_rec = coins_mesh(patternSize, cam1.rectified, cloud) # Points détectés
    if pts_rec is not None:
        ret, r, t = find_rt(patternSize, objp, cam1.not_rectified, cam1.K, cam1.D)
        rec = get_rec(objp, r, t, cam1.R, cam1.P ) # Points théoriques
        errtot, errxyz, errcyl, vecerr = err_points(patternSize, rec, pts_rec)
        errtot1.append(errtot)
        errxyz1.append(errxyz)
        errcyl1.append(errcyl)
        x0, x, y0, y, z0, z = vecerr
        X.append((x0))
        x1.append((x))
        Y.append((y0))
        y1.append((y))
        Z.append((z0))
        z1.append((z))

errz_ = []
Zmean_=[]
zmean_=[]
for i in range(len(Z)):
    zi = np.mean(z1[i])
    Zi = np.mean(Z[i])
    ei = np.absolute(Zi-zi)
    errz_.append(ei)
    Zmean_.append(Zi)
    zmean_.append(zi)


Zmean_=np.array(Zmean_)
errz_=np.array(errz_)
c=np.argsort(Zmean_)
fig, ax = plt.subplots()
ax.set_title('Erreur absolue en fonction de la distance z')
ax.set_xlabel('z (m)')
ax.set_ylabel('Erreur absolue (m)')
# plt.ylim(0,2)
plt.plot(Zmean[a], Zmean[a]-Zmean_[c], 'k.-')
plt.plot(Zmean[a], errz[a], 'r.-')
plt.plot(Zmean_[c], errz_[c], 'b.-')
plt.legend(['manufacturier', 'marianne 2'])
plt.savefig( 'output/figures/erreur_z_2.png' )
