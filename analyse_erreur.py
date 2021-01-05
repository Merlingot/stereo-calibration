
from points3d import *
from util import *

# Fichiers de calibration:
left_xml='cam1.xml'
right_xml='cam2.xml'
cam1, cam2 = get_cameras(left_xml, right_xml)

patternSize=(10,8)
squaresize=2e-2
world_th=coins_damier(patternSize,squaresize).T

N=[]
errtot1 = []
errtot2 = []
errxyz1 = []
errxyz2 = []
errcyl1 = []
errcyl2 = []

for nb in range(1,3):

    N.append(nb)
    # Images à reconstruire:
    left="captures/captures_erreur/left{}.jpg".format(nb)
    right="captures/captures_erreur/right{}.jpg".format(nb)
    cam1.set_images(left)
    cam2.set_images(right)

    # CALCUL AVEC CARTE DE DISPARITÉ  ------------------------------------------

    cloud, mask, depth_map= calcul_mesh(cam1.rectified, cam2.rectified, cam1.Q)
    save_mesh(cam1.rectified, cloud, mask, 'mesh_{}'.format(nb))
    r,t = find_rt(squaresize, patternSize, cam1.not_rectified, cam1.K, cam1.D)
    err_rt(squaresize, patternSize, cam1.not_rectified, cam1.rectified, cam1.K, cam1.D, cam1.R, cam1.P, r, t)

    rec, pts_rec, _ =coins_carte(patternSize, squaresize, r, t, cam1.R, cam1.P, cloud)
    errtot, errxyz, errcyl = err_points(patternSize, rec, pts_rec)
    errtot1.append(errtot)
    errxyz1.append(errxyz)
    errcyl1.append(errcyl)

    # CALCUL AVEC TRIANGULATION  ----------------------------------------------
    # REf monde
    wo=triangulation_world(patternSize, squaresize, cam1.K, cam2.K, cam1.D, cam2.D, cam1.not_rectified, cam2.not_rectified)
    errtot, errxyz, errcyl = err_points(patternSize, world_th, wo)
    errtot2.append(errtot)
    errxyz2.append(errxyz)
    errcyl2.append(errcyl)

    re = triangulation_rec(patternSize, cam1, cam2, cam1.not_rectified, cam2.not_rectified)

plt.plot(rec[0,:], rec[1,:], 'o')
# plt.plot(pts_rec[0,:], pts_rec[1,:], 'o')
plt.plot(re[0,:], re[1,:], 'o')

plt.plot(world_th[0,:], world_th[1,:], 'o')
plt.plot(wo[0,:], wo[1,:], 'o')

plt.imshow(depth_map)

plt.figure()
plt.title('Erreur rms sur le rayon')
plt.xlabel('# image')
plt.ylabel('Erreur (mm)')
plt.plot(N, np.array(errcyl1)[:,0]*1e3, 'bo-')
plt.plot( N, np.array(errcyl2)[:,0]*1e3, 'ro-')
plt.legend(['carte de disparité', 'triangulation'])

plt.figure()
plt.title('Erreur rms sur la position en z')
plt.xlabel('# image')
plt.ylabel('Erreur (mm)')
plt.plot(N, np.array(errxyz1)[:,2]*1e3, 'bo-')
plt.plot( N, np.array(errxyz2)[:,2]*1e3, 'ro-')
plt.legend(['carte de disparité', 'triangulation'])
