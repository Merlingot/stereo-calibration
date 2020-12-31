from points3D import *
from mesuresZ import *
from util import *

nb='15' #numero de l'image

# Fichiers de calibration:
left_xml='cam1.xml'
right_xml='cam2.xml'
# Images à reconstruire:
left="stereo/left{}.jpg".format(nb)
right="stereo/right{}.jpg".format(nb)




# CALCUL AVEC TRIANGULATION  ----------------------------------------------





# CALCUL AVEC CARTE DE DISPARITÉ  ----------------------------------------------
points, not_rectified, rectified, mask, K1, D1, R1, P1=points3D(left_xml, right_xml, left, right, i)
squaresize=3.62e-2; patternSize=(9,6)
r,t, objp, world, corners = trouver_RT(squaresize, patternSize, not_rectified, rectified, K1, D1, R1, P1 )
corners_rec=testerRT(not_rectified, rectified, patternSize, objp, world, corners, K1,D1,R1,P1,r,t)

objp=coins_damier(patternSize,squaresize); world=objp.T
unrec=r@world+t #coins théoriques dans ref cam non rectifiée
rec=R1@unrec #points théoriques dans ref cam rectifiée
corners_rec_projected, _ = cv.projectPoints(rec, np.zeros((3,1)), np.zeros((3,1)), P1[:,:3], np.zeros((1,4))) #points théoriques dans ref image cam rectifiée

# plt.plot(corners_rec[:,0,0], corners_rec[:,0,1], 'o')
# plt.plot(img_rec[:,0,0], img_rec[:,0,1], 'o')

erreur(patternSize, r, t, corners_rec_projected, rec, world, points)
# ------------------------------------------------------------------------------
