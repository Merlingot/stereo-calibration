from modules.points3d import *
from modules.util import *

# %pip install scikit-spatial
from skspatial.objects import Points, Plane, Line
# from skspatial.plotting import plot_3d
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

################################################################################
# Fichiers de calibration ------------------------------------------------------
left_xml='data/6mm/cam1_cibles.xml'
right_xml='data/6mm/cam2_cibles.xml'
# Damier -----------------------------------------------------------------------
# points_per_rows=[1,1,2,2,4,4,4,1]
# points_per_rows=[2,3,4,4,4,4,4,1]
points_per_rows=[4,3,4,4,4,4,4,1]
c = np.cumsum([0]+points_per_rows)
# Images -----------------------------------------------------------------------
left='data/6mm/cibles/left.jpg'
right='data/6mm/cibles/right.jpg'
# Coordonées des cibles
fname_cibles="data/6mm/cibles/pts_left_.txt"
fname_cibles_l="data/6mm/cibles/objpts_.txt"
################################################################################



def stats_spherique(rec, pts_rec, points_per_rows, name='spherique'):

    rows=np.cumsum([0]+points_per_rows)
    # Transformation en coordonées sphériques :
    X,Y,Z = rec[:,0], rec[:,1], rec[:,2]
    x,y,z = pts_rec[:,0], pts_rec[:,1], pts_rec[:,2]
    Xs=X; Ys=Z; Zs=-Y
    xs=x;ys=z;zs=-y

    R=np.sqrt(Xs**2+Ys**2+Zs**2)
    Phi=np.arccos(Zs/R)*180/np.pi
    Theta=np.arctan(Xs/Ys)*180/np.pi

    r=np.sqrt(xs**2+ys**2+zs**2)
    phi=np.arccos(zs/r)*180/np.pi #déviation par rapport à l'axe y
    theta=np.arctan(xs/ys)*180/np.pi # déviation par rapport à l'axe x

    devR=np.absolute(R-r)
    devPhi=np.absolute(Phi-phi)
    devTheta=np.absolute(Theta-theta)

    # Statistiques:
    stdR=np.sqrt( np.mean( devR**2 ) )
    stdPhi=np.sqrt( np.mean( devPhi**2 ) )
    stdTheta=np.sqrt( np.mean( devTheta**2 ) )


    plt.figure()
    plt.title("Erreur sur l'angle $\Theta$")
    plt.xlabel('Angle $\Theta$ (deg)')
    plt.ylabel('Erreur relative (%)')
    for i in range(len(rows)-1):
        plt.plot(np.absolute(Theta[rows[i]:rows[i+1]]), (devTheta/np.absolute(Theta)*100)[rows[i]:rows[i+1]], 'o-', label='rangée {}'.format(i+1))
    plt.legend()
    plt.savefig('{}_theta.png'.format(name))

    plt.figure()
    plt.title("Erreur sur l'angle $\phi$")
    plt.xlabel('Angle $\phi$ (deg)')
    plt.ylabel('Erreur relative (%)')
    for i in range(len(rows)-1):
        plt.plot(np.absolute(Phi[rows[i]:rows[i+1]]), (devPhi/np.absolute(Phi)*100)[rows[i]:rows[i+1]], 'o-', label='rangée {}'.format(i+1))
    plt.legend()
    plt.savefig('{}_phi.png'.format(name))

    plt.figure()
    plt.title("Erreur sur le rayon")
    plt.xlabel('Rayon $r$ (m)')
    plt.ylabel('Erreur absolue (m)')
    for i in range(len(rows)-1):
        plt.plot(R[rows[i]:rows[i+1]], devR[rows[i]:rows[i+1]], 'o-', label='rangée {}'.format(i+1))
    plt.legend()
    plt.savefig('{}_rho.png'.format(name))

    return std, (stdR, stdTheta, stdPhi)



def stats_cartesian(real, cal, points_per_rows, name):

    rows=np.cumsum([0]+points_per_rows)
    err = np.absolute(real-cal) #vecteur de distances euclidiennes
    # Statistiques:
    dev = np.linalg.norm(err, axis=1) # norme de la distance euclidienne entre chaque point
    stdx=np.sqrt( np.mean( err[:,0]**2 ) )
    stdy=np.sqrt( np.mean( err[:,1]**2 ) )
    stdz=np.sqrt( np.mean( err[:,2]**2 ) )
    std =np.sqrt( np.mean( dev**2 ) )

    plt.figure()
    plt.title('Erreur sur la coordonnée $x$')
    plt.xlabel('Coordonnée $x$ (m)')
    plt.ylabel('Erreur absolue (m)')
    for i in range(len(rows)-1):
        plt.plot(real[rows[i]:rows[i+1],0], err[rows[i]:rows[i+1],0], '-o', label='rangée {}'.format(i+1))
    plt.legend()
    plt.savefig('{}_x.png'.format(name))

    plt.figure()
    plt.title('Erreur sur la coordonnée $y$')
    plt.xlabel('Coordonnée $y$ (m)')
    plt.ylabel('Erreur absolue (m)')
    for i in range(len(rows)-1):
        plt.plot(real[rows[i]:rows[i+1],1], err[rows[i]:rows[i+1],1], '-o', label='rangée {}'.format(i+1))
    plt.legend()
    plt.savefig('{}_y.png'.format(name))


    plt.figure()
    plt.title('Erreur sur la coordonnée $z$')
    plt.xlabel('Coordonnée $z$ (m)')
    plt.ylabel('Erreur absolue (m)')
    for i in range(len(rows)-1):
        plt.plot(real[rows[i]:rows[i+1],2], err[rows[i]:rows[i+1],2], '-o', label='rangée {}'.format(i+1))
    plt.legend()
    plt.savefig('{}_z.png'.format(name))


    errtot=np.sqrt(err[:,0]**2+err[:,1]**2+err[:,2]**2)
    plt.figure()
    plt.title('Erreur en fonction de $z$')
    plt.xlabel('Coordonnée $z$ (m)')
    plt.ylabel('Erreur (m)')
    z = []; r = []
    for i in range(len(rows)-1):
        zmean = (real[rows[i]:rows[i+1],2]).mean()
        rms = np.sqrt(((err[:,2][rows[i]:rows[i+1]])**2).mean())
        plt.plot(real[rows[i]:rows[i+1],2], errtot[rows[i]:rows[i+1]], '-o', label='rangée {}'.format(i+1))
        z.append(zmean); r.append(rms)
    plt.legend()
    plt.savefig('{}_tot.png'.format(name))

    plt.figure()
    plt.title('Erreur en fonction de $z$')
    plt.xlabel('Coordonnée $z$ (m)')
    plt.ylabel('Erreur (m)')
    # for i in range(len(rows)-1):
    plt.plot(real[:,2], errtot, 'o', color='C2')
    plt.plot(z,r)
    plt.legend(['erreur absolue','erreur rms', ])
    plt.savefig('{}_alloz.png'.format(name))
    plt.show()

    # MONDE
    # errtot=np.sqrt(err[:,0]**2+err[:,1]**2+err[:,2]**2)
    # plt.figure()
    # plt.title('Erreur en fonction de $z$')
    # plt.xlabel('Coordonnée $z$ (m)')
    # plt.ylabel('Erreur (m)')
    # z = []; r = []
    # for i in range(len(rows)-1):
    #     zmean = (real[rows[i]:rows[i+1],1]).mean()
    #     rms = np.sqrt(((err[:,1][rows[i]:rows[i+1]])**2).mean())
    #     plt.plot(real[rows[i]:rows[i+1],1], errtot[rows[i]:rows[i+1]], '-o', label='rangée {}'.format(i+1))
    #     z.append(zmean); r.append(rms)
    # plt.legend()
    # plt.savefig('{}_tot_y.png'.format(name))
    #
    # plt.figure()
    # plt.title('Erreur en fonction de $z$')
    # plt.xlabel('Coordonnée $z$ (m)')
    # plt.ylabel('Erreur (m)')
    # # for i in range(len(rows)-1):
    # plt.plot(real[:,1], errtot, 'o', color='C2')
    # plt.plot(z,r)
    # plt.legend(['erreur absolue','erreur rms', ])
    # plt.savefig('{}_alloy.png'.format(name))
    # plt.show()


    return std, ( stdx, stdy,stdz)


# def erreur_cibles(left_xml, right_xml, left, right, fname_cible, fname_cible_l, points_per_rows ):

# Coordonées des cibles ----------------------------------------------------
# Référentiel caméras non rectifié
cibles_l=np.genfromtxt(fname_cibles).astype(np.float32)
cibles_l=cibles_l.reshape(cibles_l.shape[0], 1, 2)
# Référentiel monde
cibles = np.genfromtxt(fname_cibles_l).astype(np.float32)
cibles=cibles[:cibles_l.shape[0], :]
world=cibles

# CALCUL DES POINTS 3D  ----------------------------------------------------

# 1. Reconstruire l'image --------------------------------------------------
# Caméras
cam1, cam2 = get_cameras(left_xml, right_xml, alpha=-1)
# Setter les images
cam1.set_images(left)
cam2.set_images(right)

# Mesh:
cloud, mask, depth_map= calcul_mesh(cam1.rectified, cam2.rectified, cam1.Q)
# --------------------------------------------------------------------------

# 2. Get les points reconstruits -------------------------------------------
# Recitifier les coins des cibles
corners_rec=cv.undistortPoints(cibles_l, cam1.K, cam1.D, None, cam1.R, cam1.P)
# Points 3D dans le réféntiel de la caméra rectifiée
pts_rec=get_median(corners_rec, cloud, n=10)

# plt.figure()
plt.imshow(cam1.rectified)
plt.plot(corners_rec[:,0,0], corners_rec[:,0,1], 'r.')
# plt.show()


# Faire le ménage des cibles qui ne sont pas visibles :(
new_corners_rec=[]
old_pts_rec=pts_rec.copy()
pts_rec=[]
new_world=[]
new_points_per_rows=[]
for i in range(len(c)-1):
    row = old_pts_rec[c[i]:c[i+1], :]
    nb_row=0
    for j in range(len(row)):
        p = row[j]
        if not (np.isinf(p).any() or np.isnan(p).any()):
            new_corners_rec.append(corners_rec[c[i]+j])
            new_world.append(world[c[i]+j])
            pts_rec.append(p)
            nb_row+=1
    if nb_row > 0 :
        new_points_per_rows.append(nb_row)
pts_rec = np.array(pts_rec)
new_world=np.array(new_world)
new_corners_rec=np.array(new_corners_rec)


# Montrer les cibles
# Sur la carte de profondeur
plt.figure()
plt.imshow(depth_map)
plt.plot(new_corners_rec[:,0,0], new_corners_rec[:,0,1], 'r.')
plt.show()
# --------------------------------------------------------------------------

# 3. Évaluer la planéité ---------------------------------------------------
# 3.1 Trouver l'équation du plan
points = Points(pts_rec)
plane = Plane.best_fit(points)
a,b,c,d = plane.cartesian()

x = pts_rec[:,0]
y = pts_rec[:,1]
z = pts_rec[:,2]

X = np.linspace(x.min()*0.8, x.max()*1.3)
Y = np.linspace(y.min()*0.8, y.max()*1.3)
X, Y = np.meshgrid(X, Y)
Z = -(d + a*X + b*Y)/c

# Montrer le plan
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_title('Position des cibles et plan')
surf = ax.plot_surface(X, Z, -Y, linewidth=0, alpha=0.5)
ax.scatter(pts_rec[:,0], pts_rec[:,2], -pts_rec[:,1])
ax.set_xlabel('X'); ax.set_ylabel('Z'); ax.set_zlabel('Y')
plt.show()


# 3.2 Caluler l'écart-type par rapport au plan
deviations = []
for point in points:
    deviation = plane.distance_point(point)
    deviations.append(deviation)
std_plan = np.sqrt(np.mean( np.array(deviations)**2 ))
print('std plan: {}'.format(std_plan))

# --------------------------------------------------------------------------

# 4. Évaluer les distances entre les cibles  -------------------------------
distanceW=np.linalg.norm(new_world[0]-new_world, axis=1)
distanceR=np.linalg.norm(pts_rec[0]-pts_rec, axis=1)
deviations = np.array(distanceW)-np.array(distanceR)
std_distance=np.sqrt(np.mean( deviations**2 ))
deviations
print('std distance: {}'.format(std_distance))
# --------------------------------------------------------------------------

# 5. Calculer les erreurs et les stats  ------------------------------------

# RÉFÉRENTIEL MONDE
ret, rvec, t = cv.solvePnP(world, cibles_l, cam1.K, cam1.D )
r,_=cv.Rodrigues(rvec)
pts_unrec = (cam1.R.T@pts_rec.T).T
pts_world = (r.T@(pts_unrec.T - t)).T


# 3. Évaluer la planéité ---------------------------------------------------
# 3.1 Trouver l'équation du plan
points = Points(pts_world)
plane = Plane.best_fit(points)
a,b,c,d = plane.cartesian()

x = pts_world[:,0]
y = pts_world[:,1]
z = pts_world[:,2]

X = np.linspace(x.min()*0.8, x.max()*1.3)
Y = np.linspace(y.min()*0.8, y.max()*1.3)
X, Y = np.meshgrid(X, Y)
Z = -(d + a*X + b*Y)/c

# Montrer le plan
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.set_title('Position des cibles et plan')
# surf = ax.plot_surface(X, Y, Z, linewidth=0, alpha=0.5)
# ax.scatter(pts_world[:,0], pts_world[:,1], pts_world[:,2])
# ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
# plt.show()


# 3.2 Caluler l'écart-type par rapport au plan
deviations = []
for point in points:
    deviation = plane.distance_point(point)
    deviations.append(deviation)
std_plan = np.sqrt(np.mean( np.array(deviations)**2 ))
print('std plan: {}'.format(std_plan))
# --------------------------------------------------------------------------

# plt.figure()
# plt.title('Position des cibles dans le référentiel monde')
# plt.xlabel('x')
# plt.ylabel('y')
# plt.plot(new_world[:,0],new_world[:,1], '-o', label='points théoriques')
# plt.plot(pts_world[:,0],pts_world[:,1], '-o', label='points calculés')
# plt.savefig('position_cibles_world.png')
# plt.legend()

# x,y,z=pts_world[:,0],pts_world[:,1],pts_world[:,2]
# xw,yw,zw=new_world[:,0],new_world[:,1],new_world[:,2]
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.set_title('Position des cibles dans le référentiel monde')
# ax.scatter(xw, yw, zw)
# ax.scatter(x, y, z)
# ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
# plt.show()



# RÉFÉRENTIEL RECTIFIÉ
unrec=r@new_world.T+t #coins théoriques dans ref cam1 non rectifiée
rec=(cam1.R@unrec).T #points théoriques dans ref cam rectifiée

plt.figure()
plt.title('Position des cibles dans le référentiel \n rectifié de la caméra de gauche')
plt.xlabel('x')
plt.ylabel('z')
plt.plot(rec[:,0],rec[:,2], '-o', label='points théoriques')
plt.plot(pts_rec[:,0],pts_rec[:,2], '-o', label='points calculés')
plt.savefig('position_cibles_rec.png')
plt.legend()

x,y,z=pts_rec[:,0],pts_rec[:,1],pts_rec[:,2]
xw,yw,zw=rec[:,0],rec[:,1],rec[:,2]
# from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_title('Position des cibles dans le référentiel \n rectifié de la caméra de gauche')
ax.scatter(xw, zw, -yw)
ax.scatter(x, z, -y)
ax.set_xlabel('X'); ax.set_ylabel('Z'); ax.set_zlabel('Y')
plt.show()

print(std_plan)
# std, ( stdx, stdy,stdz) = stats_cartesian(new_world, pts_world, new_points_per_rows, 'monde')
# print(std)
std, ( stdx, stdy,stdz) = stats_cartesian(rec, pts_rec, new_points_per_rows, 'rectifie')
print(std)

# RÉFÉRENTIEL RECTIFIÉ - COORDONÉES SPHÉRIQUES
# _, ( stdRho, stdTheta, stdPhi) = stats_spherique(rec, pts_rec, new_points_per_rows, name='spherique')
# std_plan, std_distance, std = erreur_cibles(left_xml, right_xml, left, right, fname_cibles, fname_cibles_l, points_per_rows )
# print(std_plan, std_distance, std)
