import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import glob, os

from stereo_tools import *
from util import *


class Camera():

    def __init__(self, cameraMatrix, distCoeffs, rectificationMatix, projectionMatrix, not_rectified, rectified):

        self.K=cameraMatrix
        self.D=distCoeffs
        self.R=rectificationMatix
        self.P=projectionMatrix

        self.not_rectified=not_rectified
        self.rectified=rectified



def get_cameras(left_xml, right_xml, left, right):

    # RESOLUTION ET TAILLE -----------------------------------------------------
    resolution='VGA'
    width,height = get_image_dimension_from_resolution(resolution)
    image_size = Resolution(width,height)
    # --------------------------------------------------------------------------

    # LIRE FICHIERS DE CALIBRATION ---------------------------------------------
    P1, P2, map_left_x, map_left_y, map_right_x, map_right_y, Q, R1, K1, D1 = init_calibration(left_xml,right_xml,image_size, resolution)
    # --------------------------------------------------------------------------

    # LIRE IMAGES ORIGINALES ---------------------------------------------------
    not_rectifiedL=cv.imread(left)
    not_rectifiedR=cv.imread(right)
    # --------------------------------------------------------------------------

    # RECTIFICATION ------------------------------------------------------------
    rectifiedL,rectifiedR= get_rectified_left_right(not_rectifiedL,not_rectifiedR, map_left_x, map_left_y, map_right_x, map_right_y)
    # --------------------------------------------------------------------------

    # CAMÉRAS ------------------------------------------------------------------
    cam1 = Camera(K1,D1,R1,P1, not_rectifiedL, rectifiedL)
    cam2 = Camera(K2,D2,R2,P2, not_rectifiedR, rectifiedR)
    # --------------------------------------------------------------------------

    return cam1, cam2

def calcul_points3d(rectifiedL, rectifiedR, Q):

    # CREATION STEREO MATCHERS -------------------------------------------------
    num_disp = 5*16
    min_disp= 1
    wsize = 5
    left_matcher = cv.StereoSGBM_create(
            minDisparity=min_disp,
            numDisparities=num_disp,
            blockSize=wsize,
            P1=24*wsize*wsize,
            P2=96*wsize*wsize,
            preFilterCap=63,
            mode=cv.STEREO_SGBM_MODE_SGBM_3WAY)
    right_matcher = cv.ximgproc.createRightMatcher(left_matcher)
    # --------------------------------------------------------------------------

    # FORMATAGE PRE-CALCUL -----------------------------------------------------
    # DOWNSCALE
    downscale=1
    new_num_disp = int(num_disp / downscale)
    n_width = int(rectifiedL.shape[1] * 1/downscale)
    n_height = int(rectifiedR.shape[0] * 1/downscale)
    downL = cv.resize(rectifiedL, (n_width, n_height))
    downR = cv.resize(rectifiedR, (n_width, n_height))

    # GRAY SCALE
    grayL_down = cv.cvtColor(downL,cv.COLOR_BGR2GRAY)
    grayR_down = cv.cvtColor(downR,cv.COLOR_BGR2GRAY)

    # SMOOTH
    # grayL_down = cv.medianBlur(grayL_down,3)
    # grayR_down = cv.medianBlur(grayR_down,3)
    # --------------------------------------------------------------------------

    # CALCULS DISPARITÉ --------------------------------------------------------
    displ = left_matcher.compute(cv.UMat(grayL_down),cv.UMat(grayR_down))
    dispr = right_matcher.compute(cv.UMat(grayR_down),cv.UMat(grayL_down))
    displ = cv.UMat.get(displ)
    dispr = cv.UMat.get(dispr)
    # --------------------------------------------------------------------------

    # WLS FILTER ---------------------------------------------------------------
    lambda_wls = 8000.0
    sigma_wls = 1.5
    wls_filter = cv.ximgproc.createDisparityWLSFilter(matcher_left=left_matcher)
    wls_filter.setLambda(lambda_wls)
    wls_filter.setSigmaColor(sigma_wls)
    filtered_disp = wls_filter.filter(displ, rectifiedL, None, dispr)
    # --------------------------------------------------------------------------

    # MASK ---------------------------------------------------------------------
    # CONFIDENCE MAP
    conf_map=wls_filter.getConfidenceMap()
    # ROI
    ROI = np.array(wls_filter.getROI())*downscale
    #MASK
    mask=np.zeros(conf_map.shape, conf_map.dtype)
    mask[ROI[1]:ROI[1]+ROI[3], ROI[0]:ROI[0]+ROI[2]]=1
    # --------------------------------------------------------------------------

    # BILATERAL FILTER ---------------------------------------------------------
    fbs_spatial=80.0
    fbs_luma=8.0
    fbs_chroma=8.0
    fbs_lambda=128.0
    solved_filtered_disp = cv.ximgproc.fastBilateralSolverFilter(rectifiedL, filtered_disp, conf_map/255.0, None, fbs_spatial, fbs_luma, fbs_chroma, fbs_lambda)
    # --------------------------------------------------------------------------

    # REPROJECT TO 3D ----------------------------------------------------------
    # format disparity
    disparity=solved_filtered_disp.astype(np.float32)/16.0*mask

    # Note: If one uses Q obtained by stereoRectify, then the returned points are represented in the first camera's rectified coordinate system
    points = (-1)*cv.reprojectImageTo3D(disparity, Q, handleMissingValues=True)

    # depth map
    depth_map=points[:,:,2]*mask
    # plt.figure()
    # plt.imshow(depth_map)
    # plt.savefig('depth_map_{}.png'.format(nb))
    # -------------------------------------------------------------------------

    return points, mask, depth_map


def save_mesh(color_image, points, mask, fname):

    # SAVEGARDER MESH ----------------------------------------------------------
    colors = cv.cvtColor(color_image, cv.COLOR_BGR2RGB)
    colors_valides = colors[mask.astype(bool)]
    points_valides=points[mask.astype(bool)]
    out_fn = '3dpoints/{}.ply'.format(fname)
    write_ply(out_fn, points_valides, colors_valides)
    # --------------------------------------------------------------------------


def find_rt(squaresize, patternSize, not_rectified, rectified, K, D):
    """
    Trouver la transformation (r,t) qui amène du référentiel de la caméra 1 rectifiée (Xc,Yc,Zc)_rec au référentiel monde (Xw, Yw,Zw)

    R1: performs a change of basis from the unrectified first camera's coordinate system to the rectified first camera's coordinate system.
    P1 : projects points given in the rectified first camera coordinate system into the rectified first camera's image
    r,t : performs a change of basis form the world coordinate system to the first camera's distorted and unrectifed coordinate system

    Coordonées cam rectifiée: rec=(Xc,Yc,Zc)_rec
    Coordonées cam non-rectifiée : unrec=R1.T@rec=(Xc,Yc,Zc)
    Coordonnées world : world=r.T@(unrec-t1)=(Xw,Yw,Zw)
    """

    # Coordonées des coins du damier dans le référentiel monde:
    objp=coins_damier(patternSize, squaresize)
    world=objp.T

    # DÉTECTION DES COINS DANS L'IMAGE NON RECTIFIÉE ---------------------------
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    # Trouver coins dans l'image originale prise par la caméra
    gray=cv.cvtColor(not_rectified, cv.COLOR_BGR2GRAY)
    ret, corners_unrec = cv.findChessboardCorners(gray, patternSize, None)
    if ret :
        corners_unrec = cv.cornerSubPix(gray, corners_unrec, (11, 11),(-1, -1), criteria)
    assert ret==True, "coins non détectés"
    # --------------------------------------------------------------------------

    # RÉSOLUTION POUR TROUVER R,T (unrec=R@world+T) ----------------------------
    ret, rvec, t = cv.solvePnP(objp, corners_unrec, K, D )
    r=cv.Rodrigues(rvec)[0]
    # --------------------------------------------------------------------------

    return r, t


def test_rt(squaresize, patternSize, not_rectified, rectified, K, D, R, P, r, t):

    # DÉTECTION DES COINS DANS L'IMAGE NON RECTIFIÉE ---------------------------
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    # Trouver coins dans l'image originale prise par la caméra
    gray=cv.cvtColor(not_rectified, cv.COLOR_BGR2GRAY)
    ret, corners_unrec = cv.findChessboardCorners(gray, patternSize, None)
    if ret :
        corners_unrec = cv.cornerSubPix(gray, corners_unrec, (11, 11),(-1, -1), criteria)
    assert ret==True, "coins non détectés"
    # --------------------------------------------------------------------------

    # DÉTECTION DES COINS DANS L'IMAGE RECTIFIÉE -------------------------------
    # On détecte les coins dans l'image de gauche rectifiée.
    gray=cv.cvtColor(rectified, cv.COLOR_BGR2GRAY)
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    ret, corners_rec = cv.findChessboardCorners(gray, patternSize, None)
    if ret:
        corners_rec = cv.cornerSubPix(gray, corners_rec, (11, 11),(-1, -1), criteria)
    assert ret==True, "coins non détectés"
    # --------------------------------------------------------------------------


    # PROJECTION DES COORD MONDE -----------------------------------------------
    # Coordonées des coins du damier dans le référentiel monde:
    objp=coins_damier(patternSize, squaresize)
    world=objp.T
    # Rodrigues
    rvec, _ =cv.Rodrigues(r)

    # ->Référentiel image de la caméra non rectifiée
    img_unrec, _=cv.projectPoints(world, rvec, t, K, D )

    # ->Référentiel image de la caméra rectifiée
    unrec=r@world+t #coins théoriques dans ref cam non rectifiée
    rec=R@unrec #points théoriques dans ref cam rectifiée
    img_rec, _ = cv.projectPoints(rec, np.zeros((3,1)), np.zeros((3,1)), P[:,:3], np.zeros((1,4))) #points théoriques dans ref image cam rectifiée
    # --------------------------------------------------------------------------

    # COMPARAISON DES POINTS DÉTECTÉS ET CEUX PROJETÉS -------------------------
    # Référentiel non rectifié
    err_unrec = np.sqrt( np.mean( np.sum( (corners_unrec-img_unrec)**2, axis=2 )  )  )
    # Référentiel rectifié
    err_rec = np.sqrt( np.mean( np.sum( (corners_rec-img_rec)**2, axis=2 ) )  )

    print('erreurs de reprojection: référentiels non rectifié et rectifié')
    print(err_unrec, err_rec)
    # --------------------------------------------------------------------------

def erreur(patternSize, r, t, corners_rec, rec, world, points):

    # COMPRAISON DES COINS DANS LA CARTE DE PROFONDEUR -------------------------
    # 1. Référentiel caméra 1 rectifiée -----------
    # On prend les coins images et on trouve l'équivalent 3D calculé
    pts_rec=[]
    for i in range(len(corners_rec)):
        col, row = int(np.round(corners_rec[i,0,0])), int(np.round(corners_rec[i,0,1]))
        pt = points[row,col]
        pts_rec.append(pt)
    pts_rec=np.array(pts_rec).T #points 3D dans le réféntiel de la caméra rectifiée


    # ERREUR -------------------------------------------------------------------
    arr=np.zeros((patternSize[1],patternSize[0]))
    x,y,z=arr.copy(), arr.copy(), arr.copy()
    xo,yo,zo=arr.copy(), arr.copy(), arr.copy()
    j=0;n=9
    for i in range(patternSize[1]):
        x[i,:]=pts_rec[0,j:j+n]; y[i,:]=pts_rec[1,j:j+n]; z[i,:]=pts_rec[2,j:j+n]
        xo[i,:]=rec[0,j:j+n]; yo[i,:]=rec[1,j:j+n]; zo[i,:]=rec[2,j:j+n]
        j+=n

    errX=np.absolute(xo-x); errY=np.absolute(yo-y); errZ=np.absolute(zo-z)
    errRMS=np.sqrt( (errX**2 + errY**2 + errZ**2).mean() )
    return errRMS










# ############ MESH #######################################################
#
# out_points=points[mask.astype(bool)]
# # Coordonnées cam 1 rectifiée
# colors = cv.cvtColor(rectified, cv.COLOR_BGR2RGB)
# out_colors = colors[mask.astype(bool)]
# rec_points = out_points.T
# out_fn = '3dpoints/out_rec_{}.ply'.format(nb)
# write_ply(out_fn, rec_points.T, out_colors)
#
# # Coordonnées cam 1 non-rectifiée
# colors = cv.cvtColor(not_rectified, cv.COLOR_BGR2RGB)
# out_colors = colors[mask.astype(bool)]
# unrec_points = R1.T@rec_points
# out_fn = '3dpoints/out_unrec_{}.ply'.format(nb)
# write_ply(out_fn, unrec_points.T, out_colors)
#
# # Coordonnées monde
# colors = cv.cvtColor(not_rectified, cv.COLOR_BGR2RGB)
# out_colors = colors[mask.astype(bool)]
# world_points = r.T@(unrec_points-t)
# out_fn = '3dpoints/out_world_{}.ply'.format(nb)
# write_ply(out_fn, world_points.T, out_colors)

# normalized = (255-cv.normalize(depth_map,  None, 0, 255, cv.NORM_MINMAX)).astype(np.uint8)
