from modules.StereoCalibration import StereoCalibration
from modules.util import *
import cv2 as cv
import matplotlib.pyplot as plt

# ================ PARAMETRES ========================
patternSize=(15,10)
squaresize=7e-2
single_path='data/12mm/damier/tout/'
stereo_path='data/12mm/damier/tout/'
single_detected_path='output/singles_detected/'
stereo_detected_path='output/stereo_detected/'
# ====================================================

cibles=np.genfromtxt("data/12mm/cibles/objpts.txt").astype(np.float32)
cibles_l=np.genfromtxt("data/12mm/cibles/pts_left.txt").astype(np.float32)
cibles_l=cibles_l.reshape(cibles_l.shape[0], 1, 2)
cibles_r=np.genfromtxt("data/12mm/cibles/pts_right.txt").astype(np.float32)
cibles_r=cibles_r.reshape(cibles_r.shape[0], 1, 2)
cibles=cibles[:cibles_r.shape[0], :]


obj = StereoCalibration(patternSize, squaresize)
obj.calibrateIntrinsics(single_path, single_detected_path, cibles=cibles, cibles_l=cibles_l, cibles_r=cibles_r, fisheye=False)
obj.draw=True
obj.calibrateExtrinsics(stereo_path, stereo_detected_path, cibles=cibles, cibles_l=cibles_r, cibles_r=cibles_r, fisheye=False, calib_2_sets=True)
obj.saveResultsXML(left_name='data/12mm/cam1_cibles', right_name='data/12mm/cam2_cibles')
# obj.perViewErrors
# obj.reprojection('output/reprojection/',5)


import glob
def flip_images(src, dst, view, flip=-1, ext='jpg'):
    s = src + view + '*.' + 'jpg'
    g0 = np.sort(glob.glob(s))
    assert g0 >0 , 'Dossier vide: \n {}'.format(s)




g0 = glob.glob('data/12mm/cibles/originales/left*.jpg')

for i in range(len(g0)):
    img = cv.imread(g0[i])
    img=cv.flip(img,-1)
    cv.imwrite('data/12mm/cibles/flip/left{:03d}.jpg'.format(i+1),img)

g0 = glob.glob('data/12mm/damier/originales/gauche/right*.jpg')
g0 += glob.glob('data/12mm/damier/originales/droite/right*.jpg')
g0 = np.sort(g0)
for i in range(len(g0)):
    img = cv.imread(g0[i])
    cv.imwrite('data/12mm/damier/tout/right{:03d}.jpg'.format(i),img)
