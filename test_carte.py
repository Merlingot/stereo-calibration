from modules.util import *
from modules.points3d import *
import cv2 as cv

################################################################################
# Choisir une image à analyser:
left='captures/captures_calibration/left001.jpg'
right='captures/captures_calibration/right001.jpg'


# Fichiers de calibration:
left_xml='cam1.xml'
right_xml='cam2.xml'
################################################################################


# Calculs ----------------------------------------------------------------------
cam1,cam2=get_cameras(left_xml, right_xml, alpha=0)
cam1.set_images(left)
cam2.set_images(right)

cloud, mask, depth_map = calcul_mesh(cam1.rectified, cam2.rectified, cam1.Q, downscale=2, sigma_wls=1.5, fbs_spatial=8.0, bilateral_on=False)
# ------------------------------------------------------------------------------

# Montrer la carte de profondeur -----------------------------------------------
depth_map=cloud[:,:,2]*mask
fig, ax=plt.subplots()
a = plt.imshow(depth_map)
a.set_clim(0,35)
plt.colorbar()
plt.show()
# ------------------------------------------------------------------------------

# Sauvegarder le mesh ----------------------------------------------------------
colors = cv.cvtColor(cam1.rectified, cv.COLOR_BGR2RGB)
colors_valides = colors[mask.astype(bool)]
points_valides=cloud[mask.astype(bool)]
out_fn = 'output/3dpoints/{}.ply'.format('my_mesh')
write_ply(out_fn, points_valides, colors_valides)
# ------------------------------------------------------------------------------
