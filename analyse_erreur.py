
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


z1=[]
ez1=[]
z2=[]
ez2=[]

r1=[]
er1=[]
r2=[]
er2=[]

t1=[]
et1=[]
t2=[]
et2=[]

# Damier et caméras ------------------------------------------------------------
objp = coins_damier(patternSize,squaresize)
world = objp.T
cam1, cam2 = get_cameras(left_xml, right_xml)
# ------------------------------------------------------------------------------

left=np.sort(glob.glob("captures/captures_test/left*.jpg"))
right=np.sort(glob.glob("captures/captures_test/right*.jpg"))


for nb in range(len(left)):

    N.append(nb)
    # Images à reconstruire:
    cam1.set_images(left[nb])
    cam2.set_images(right[nb])

    # CALCUL AVEC CARTE DE DISPARITÉ  ------------------------------------------
    cloud, mask, depth_map= calcul_mesh(cam1.rectified, cam2.rectified, cam1.Q)
    save_mesh(cam1.rectified, cloud, mask, 'mesh_{}'.format(nb))
    pts_rec, corners_rec = coins_mesh(patternSize, cam1.rectified, cloud) # Points détectés

    ret, r, t = find_rt(patternSize, objp, cam1.not_rectified, cam1.K, cam1.D)
    rec = get_rec(objp, r, t, cam1.R, cam1.P ) # Points théoriques

    errtot, errxyz, errcyl, vecerr = err_points(patternSize, rec, pts_rec)
    errtot1.append(errtot)
    errxyz1.append(errxyz)
    errcyl1.append(errcyl)
    vecz, ez, vecr,er, vect, et = vecerr
    z1.append((vecz))
    ez1.append((ez))
    r1.append((vecr))
    er1.append((er))

    # --------------------------------------------------------------------------

    # CALCUL AVEC TRIANGULATION  ----------------------------------------------
    re = triangulation_rec(patternSize, cam1.rectified, cam2.rectified, cam1.P, cam2.P ) # Points détectés
    errtot, errxyz, errcyl, vecerr = err_points(patternSize, rec, re)
    errtot2.append(errtot)
    errxyz2.append(errxyz)
    errcyl2.append(errcyl)
    vecz, ez, vecr,er, vect, et = vecerr

    ez2.append((ez))
    er2.append((er))

    # --------------------------------------------------------------------------


fig, ax = plt.subplots()
ax.set_title('Erreur absolue en fonction du rayon r')
ax.set_xlabel('Rayon (mm)')
ax.set_ylabel('Erreur absolue (m)')
for i in range(len(r1)):
    plt.plot(np.array(r1)[i]*1e3, np.array(er1)[i], 'o-')
    # plt.plot(np.array(r1)[i]*1e3, np.array(er2)[i]*1e3, 'o-')
# plt.legend(['carte', 'triangulation'])
plt.savefig( 'erreur_rabs1.png' )

fig, ax = plt.subplots()
ax.set_title('Erreur absolue en fonction du rayon r')
ax.set_xlabel('Rayon (mm)')
ax.set_ylabel('Erreur absolue (mm)')
for i in range(len(r1)):
    # plt.plot(np.array(r1)[i]*1e3, np.array(er1)[i]*1e3, 'o-')
    plt.plot(np.array(r1)[i]*1e3, np.array(er2)[i]*1e3, 'o-')
# plt.legend(['carte', 'triangulation'])
plt.savefig( 'erreur_rabs2.png' )


fig, ax = plt.subplots()
ax.set_title('Erreur absolue en fonction de la ditance z')
ax.set_xlabel('z (m)')
ax.set_ylabel('Erreur absolue (mm)')
for j in range(len(z1)):
    plt.plot(np.array(z1)[j], np.array(ez1)[j]*1e3, 'o-')
    # plt.plot(np.array(z1)[j], np.array(ez2)[j]*1e3, 'o-')
# plt.legend(['carte', 'triangulation'])
plt.savefig( 'erreur_zabs1.png' )

fig, ax = plt.subplots()
ax.set_title('Erreur absolue en fonction de la ditance z')
ax.set_xlabel('z (m)')
ax.set_ylabel('Erreur absolue (mm)')
for j in range(len(z1)):
    # plt.plot(np.array(z1)[j], np.array(ez1)[j]*1e3, 'o-')
    plt.plot(np.array(z1)[j], np.array(ez2)[j]*1e3, 'o-')
# plt.legend(['carte', 'triangulation'])
plt.savefig( 'erreur_zabs2.png' )

#
# fig, ax = plt.subplots()
# ax.set_title('Erreur rms sur la position')
# ax.set_xlabel('# image')
# ax.set_ylabel('Erreur (mm)')
# ax.plot(N, np.array(errtot1)*1e3,'o-', color='tab:blue', label='carte')
# ax.plot( N, np.array(errtot2)*1e3,'o-', color='tab:orange', label='triangulation')
# plt.legend()
# plt.savefig( 'erreur_tot.png' )



# fig, ax1 = plt.subplots()
# ax1.set_title('Erreur rms sur le rayon (en x-y)')
# ax1.set_xlabel('# image')
# ax1.set_ylabel('Erreur carte de disparité (mm)')
# ax1.tick_params(axis='y', labelcolor='tab:blue')
# ax1.plot(N, np.array(errcyl1)[:,0]*1e3,'o-', color='tab:blue')
# ax2=ax1.twinx()
# ax2.tick_params(axis='y', labelcolor='tab:orange')
# ax2.set_ylabel('Erreur triangulation (mm)')
# ax2.plot( N, np.array(errcyl2)[:,0]*1e3,'o-', color='tab:orange')
# plt.savefig( 'erreur_rayon.png' )
#
# fig, ax1 = plt.subplots()
# ax1.set_title('Erreur rms sur la position en z en millimètres')
# ax1.set_xlabel('# image')
# ax1.set_ylabel('Erreur -méthode carte de disparité- (mm)', color='tab:blue')
# ax1.tick_params(axis='y', labelcolor='tab:blue')
# ax1.plot(N, np.array(errxyz1)[:,0]*1e3,'o-', color='tab:blue')
# ax2=ax1.twinx()
# ax2.set_ylabel('Erreur -méthode triangulation- (mm)', color='tab:orange')
# ax2.plot( N, np.array(errxyz2)[:,0]*1e3, 'o-', color='tab:orange')
# ax2.tick_params(axis='y', labelcolor='tab:orange')
# plt.savefig( 'erreur_z.png' )



# plt.figure()
# plt.title('Reconstruction plan x-y')
# plt.xlabel('x')
# plt.ylabel('y')
# plt.plot(rec[0,:], rec[1,:], 'o-') # théorique
# plt.plot(re[0,:], re[1,:], '.-') # théorique
# plt.plot(pts_rec[0,:], pts_rec[1,:], '.-') # détecté
# plt.legend(['points théoriques', 'triangulation', 'carte'])
# plt.savefig('xy_27.png')
#
# plt.figure()
# plt.title('Reconstruction plan x-z')
# plt.xlabel('x')
# plt.ylabel('z')
# plt.plot(rec[0,:], rec[2,:], 'o-')
# plt.plot(re[0,:], re[2,:], '.-') # théorique
# plt.plot(pts_rec[0,:], pts_rec[2,:], '.-')
# plt.legend(['points théoriques', 'triangulation', 'carte'])
# plt.savefig('xz_27.png')
#
# plt.figure()
# plt.title('Reconstruction plan y-z')
# plt.xlabel('y')
# plt.ylabel('z')
# plt.plot(rec[1,:], rec[2,:], 'o-')
# plt.plot(re[1,:], re[2,:], '.-') # théorique
# plt.plot(pts_rec[1,:], pts_rec[2,:], '.-')
# plt.legend(['points théoriques', 'triangulation', 'carte'])
# plt.savefig('yz_27.png')
