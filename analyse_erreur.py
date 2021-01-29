
from modules.points3d import *
from modules.util import *

################################################################################
# Fichiers de calibration ------------------------------------------------------
left_xml='cam1_zed.xml'
right_xml='cam2_zed.xml'
# Damier -----------------------------------------------------------------------
patternSize=(15,10)
squaresize=7e-2
# Images -----------------------------------------------------------------------
left=np.concatenate( (np.sort(glob.glob("captures_zed/captures_3/left*.jpg"))[0:15] , np.sort(glob.glob("captures_zed/captures_2/left*.jpg"))[6:14]))
left
right=np.concatenate( (np.sort(glob.glob("captures_zed/captures_3/right*.jpg"))[0:15] , np.sort(glob.glob("captures_zed/captures_2/right*.jpg"))[6:14]))
################################################################################

# Déclaration des listes d'erreur ----------------------------------------------
N=[]
errtot1 = []; errtot2 = []
# errxyz1 = []; errxyz2 = []
# errcyl1 = []; errcyl2 = []
Z=[];z1=[];z2=[]
X=[];x1=[];x2=[]
Y=[];y1=[];y2=[]
images=[]
# Damier et caméras ------------------------------------------------------------
objp = coins_damier(patternSize,squaresize)
world = objp.T
cam1, cam2 = get_cameras(left_xml, right_xml, alpha=0)
# ------------------------------------------------------------------------------

for nb in range(len(left)):

    N.append(nb)
    # Images à reconstruire:
    cam1.set_images(left[nb])
    cam2.set_images(right[nb])
    images.append(cam1.rectified)

    # CALCUL AVEC CARTE DE DISPARITÉ  ------------------------------------------
    cloud, mask, depth_map= calcul_mesh(cam1.rectified, cam2.rectified, cam1.Q)
    # save_mesh(cam1.rectified, cloud, mask, 'mesh_{}'.format(nb))
    pts_rec, corners_rec = coins_mesh(patternSize, cam1.rectified, cloud, winSize=(3,3)) # Points détectés

    if pts_rec is not None:
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

    # # CALCUL AVEC TRIANGULATION  ----------------------------------------------
    # re = triangulation_rec(patternSize, cam1.rectified, cam2.rectified, cam1.P, cam2.P ) # Points détectés
    # errtot, errxyz, errcyl, vecerr = err_points(patternSize, rec, re)
    # errtot2.append(errtot)
    # _, x, _, y, _, z = vecerr
    # x2.append((x))
    # y2.append((y))
    # z2.append((z))
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

# plt.plot(Zmean[a][errz[a]<1], errz[a][errz[a]<1], 'r.-')

# Fichiers de calibration ------------------------------------------------------
left_xml='cam1_cibles.xml'
right_xml='cam2_cibles.xml'
# ------------------------------------------------------------------------------
# Damier et caméras ------------------------------------------------------------
cam1, cam2 = get_cameras(left_xml, right_xml, alpha=0)
# ------------------------------------------------------------------------------
# Déclaration des listes d'erreur ----------------------------------------------
N=[]
errtot1 = []; errtot2 = []
errxyz1 = []; errxyz2 = []
errcyl1 = []; errcyl2 = []
Z=[];z1=[];z2=[]
X=[];x1=[];x2=[]
Y=[];y1=[];y2=[]
images=[]


for nb in range(len(left)):

    N.append(nb)
    # Images à reconstruire:
    cam1.set_images(left[nb])
    cam2.set_images(right[nb])
    images.append(cam1.rectified)

    # CALCUL AVEC CARTE DE DISPARITÉ  ------------------------------------------
    cloud, mask, depth_map= calcul_mesh(cam1.rectified, cam2.rectified, cam1.Q)
    # save_mesh(cam1.rectified, cloud, mask, 'mesh_{}'.format(nb))
    pts_rec, corners_rec = coins_mesh(patternSize, cam1.rectified, cloud) # Points détectés
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
f=(errz[a])<2
ff=(errz_[c])<2

fig, ax = plt.subplots()
ax.set_title('Erreur absolue en fonction de la distance z')
ax.set_xlabel('z (m)')
ax.set_ylabel('Erreur absolue (m)')
# plt.ylim(0,2)
# plt.plot(Zmean[a], Zmean[a]-Zmean_[c], 'k.-')
plt.plot(Zmean[a][ff], errz[a][ff], 'r.-')
plt.plot(Zmean_[c][ff], errz_[c][ff], 'b.-')
plt.legend(['manufacturier', 'cibles'])
plt.savefig( 'output/figures/erreur_z_cibles.png' )
