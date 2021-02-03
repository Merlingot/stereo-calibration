
from modules.points3d import *
from modules.util import *

################################################################################
# Fichiers de calibration ------------------------------------------------------
left_xml='data/zed/cam1_zed.xml'
right_xml='data/zed/cam2_zed.xml'
# Damier -----------------------------------------------------------------------
patternSize=(15,10)
squaresize=7e-2
# Images -----------------------------------------------------------------------
left=np.concatenate( (np.sort(glob.glob("data/zed/damier/captures_3/left*.jpg"))[0:15] , np.sort(glob.glob("data/zed/damier/captures_2/left*.jpg"))[6:14]))
left
right=np.concatenate( (np.sort(glob.glob("data/zed/damier/captures_3/right*.jpg"))[0:15] , np.sort(glob.glob("data/zed/damier/captures_2/right*.jpg"))[6:14]))
################################################################################

# Déclaration des listes d'erreur ----------------------------------------------
N=[]
# Damier et caméras ------------------------------------------------------------
objp = coins_damier(patternSize,squaresize)
world = objp.T
cam1, cam2 = get_cameras(left_xml, right_xml, alpha=0)
# ------------------------------------------------------------------------------

err=[]
z = []

for nb in range(len(left)):

    # Images à reconstruire:
    cam1.set_images(left[nb])
    cam2.set_images(right[nb])

    # Détection du damier dans l'image non rectifiée gauche


    # CALCUL AVEC CARTE DE DISPARITÉ  ------------------------------------------
    cloud, mask, depth_map= calcul_mesh(cam1.rectified, cam2.rectified, cam1.Q)


    pts_rec, corners_rec = coins_mesh(patternSize, cam1.rectified, cloud, winSize=(3,3)) # Points détectés

    if pts_rec is not None:
        ret, r, t = find_rt(patternSize, objp, cam1.not_rectified, cam1.K, cam1.D, winSize=(3,3))
        rec = get_rec(objp, r, t, cam1.R, cam1.P )

        e, zmean = err_points(patternSize, rec, pts_rec)
        err.append(e)
        z.append(zmean)


plt.plot(z, err, 'o')
