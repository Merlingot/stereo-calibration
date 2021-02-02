from modules.points3d import *
from modules.util import *

# %pip install scikit-spatial
from skspatial.objects import Points, Plane, Line
from skspatial.plotting import plot_3d


################################################################################
# Fichiers de calibration ------------------------------------------------------
left_xml='data/zed/cam1_cibles.xml'
right_xml='data/zed/cam2_cibles.xml'
# Damier -----------------------------------------------------------------------
points_per_rows=[4,5,5]
# Images -----------------------------------------------------------------------
left='data/zed/cibles/left1.jpg'
right='data/zed/cibles/right1.jpg'
# Coordonées des cibles
fname_cibles="data/zed/cibles/pts_left1.txt"
fname_cibles_l="data/zed/cibles/objpts.txt"
################################################################################


def erreur_cibles(left_xml, right_xml, left, right, fname_cible, fname_cible_l, points_per_rows ):

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
    cam1, cam2 = get_cameras(left_xml, right_xml, alpha=0)
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

    # # Montrer les cibles
    # x,y,z=pts_rec[:,0],pts_rec[:,1],pts_rec[:,2]
    # from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(x, z, -y)
    # ax.set_xlabel('X'); ax.set_ylabel('Z'); ax.set_zlabel('Y')
    # plt.show()
    # --------------------------------------------------------------------------


    # 3. Évaluer la planéité ---------------------------------------------------
    # 3.1 Trouver l'équation du plan
    points = Points(pts_rec)
    plane = Plane.best_fit(points)
    # Montrer le plan
    # plot_3d( points.plotter(c='k', s=5, depthshade=False),
    #     plane.plotter(alpha=0.2,lims_x=(-2, 3.5), lims_y=(0, 6)))
    # plt.show()
    # 3.2 Caluler l'écart-type par rapport au plan
    deviations = []
    for point in points:
        deviation = plane.distance_point(point)
        deviations.append(deviation)
    std_plan = np.sqrt(np.mean( np.array(deviations)**2 ))
    # --------------------------------------------------------------------------

    # 4. Évaluer les distances entre les cibles  -------------------------------
    distanceW=np.linalg.norm(world[0]-world, axis=1)
    distanceR=np.linalg.norm(pts_rec[0]-pts_rec, axis=1)
    deviations = np.array(distanceW)-np.array(distanceR)
    std_distance=np.sqrt(np.mean( deviations**2 ))
    # --------------------------------------------------------------------------

    # 5. Calculer les erreurs et les stats  ------------------------------------

    # RÉFÉRENTIEL MONDE
    ret, rvec, t = cv.solvePnP(world, cibles_l, cam1.K, cam1.D )
    r,_=cv.Rodrigues(rvec)
    pts_unrec = (cam1.R.T@pts_rec.T).T
    pts_world = (r.T@(pts_unrec.T - t)).T

    plt.figure()
    plt.title('Position des cibles dans le référentiel monde')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.plot(world[:,0],world[:,1], '-o', label='points théoriques')
    plt.plot(pts_world[:,0],pts_world[:,1], '-o', label='points calculés')
    plt.legend()


    # RÉFÉRENTIEL RECTIFIÉ
    unrec=r@world.T+t #coins théoriques dans ref cam1 non rectifiée
    rec=(cam1.R@unrec).T #points théoriques dans ref cam rectifiée

    std, ( stdx, stdy,stdz) = stats_cartesian(rec, pts_rec, points_per_rows, 'rectifie')

    # RÉFÉRENTIEL RECTIFIÉ - COORDONÉES SPHÉRIQUES
    _, ( stdRho, stdTheta, stdPhi) = stats_spherique(rec, pts_rec, points_per_rows, name='spherique')
    # --------------------------------------------------------------------------

    return std_plan, std_distance, std




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


    errtot=np.sqrt( err[:,0]**2+err[:,1]**2+err[:,2]**2 ) #somme en quadrature
    plt.figure()
    plt.title('Erreur totale en fonction de $z$')
    plt.xlabel('Coordonnée $z$ (m)')
    plt.ylabel('Erreur absolue (m)')
    for i in range(len(rows)-1):
        plt.plot(real[rows[i]:rows[i+1],2], errtot[rows[i]:rows[i+1]], '-o', label='rangée {}'.format(i+1))
    plt.legend()
    plt.savefig('{}_tot.png'.format(name))

    return std, (stdx, stdy, stdz)


std_plan, std_distance, std = erreur_cibles(left_xml, right_xml, left, right, fname_cibles, fname_cibles_l, points_per_rows )
print(std_plan, std_distance, std)
