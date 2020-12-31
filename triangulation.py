import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import glob, os

from util import readXML, find_corners, read_images, write_ply, refine_corners

def triangulation(patternSize, squaresize, left_xml, right_xml, left, right, nb):

    # LIRE FICHIERS DE CALIBRATION ---------------------------------------------
    K1,D1,_,_,imageSize1, E, F=readXML(left_xml) #left
    K2,D2,R, T,imageSize2, _, _=readXML(right_xml) #right
    # La taille de la caméra de gauche est celle de référence
    width1, height1=imageSize1[1],imageSize1[0]
    width2, height2=imageSize2[1],imageSize2[0]
    # --------------------------------------------------------------------------

    # RECTIFICATION ------------------------------------------------------------
    R1, R2, P1, P2, Q , _, _= cv.stereoRectify(K1, D1, K2, D2, (height1, width1), R, T, flags=0, alpha=0)
    # --------------------------------------------------------------------------

    # LIRE IMAGES ET TROUVER POINTS --------------------------------------------
    _, grayl=read_images(left)
    _, grayr=read_images(right)
    # damier
    objp = np.zeros((patternSize[0]*patternSize[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:patternSize[0], 0:patternSize[1]].T.reshape(-1, 2)
    objp*=squaresize
    # trouver
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    ret_l, corners_l = cv.findChessboardCorners(grayl, patternSize, None)
    ret_r, corners_r = cv.findChessboardCorners(grayr, patternSize, None)
    if ret_l*ret_r :
        pts_l= cv.cornerSubPix(grayl, corners_l, (11, 11),(-1, -1), criteria)
        pts_r= cv.cornerSubPix(grayr, corners_r, (11, 11),(-1, -1), criteria)
    # --------------------------------------------------------------------------


    # MATRICES DE PROJECTION ---------------------------------------------------
    ret, rvec1, t1 = cv.solvePnP(objp, pts_l, K1, D1 )
    ret, rvec2, t2 = cv.solvePnP(objp, pts_r, K2, D2 )
    r1=cv.Rodrigues(rvec1)[0]; r2=cv.Rodrigues(rvec2)[0]
    p1=np.column_stack((r1,t1)); Proj1=K1@p1
    p2=np.column_stack((r2,t2)); Proj2=K2@p2
    # --------------------------------------------------------------------------

    # UNDISTORT POINTS ---------------------------------------------------------
    ptsl=cv.undistortPoints(pts_l, K1, D1, None, None, K1) #not rectification
    ptsr=cv.undistortPoints(pts_r, K2, D2, None, None, K2) #not rectification
    N=ptsl.shape[0]
    projPoints1 = np.zeros((2,N))
    projPoints2 = np.zeros((2,N))
    for i in range(N):
        projPoints1[:,i]=ptsl[i,0]
        projPoints2[:,i]=ptsr[i,0]
    # --------------------------------------------------------------------------

    # TRIANGULATION ------------------------------------------------------------
    #triangulatePoints: If the projection matrices from stereoRectify are used, then the returned points are represented in the first camera's rectified coordinate system.
    points4D=cv.triangulatePoints(Proj1, Proj2, projPoints1, projPoints2)
    points3D=cv.convertPointsFromHomogeneous(points4D.T)

    X,Y,Z=points3D[:,0,0], points3D[:,0,1], points3D[:,0,2]
    # --------------------------------------------------------------------------

    # CALCUL DES ERREURS -------------------------------------------------------
    patternSizeT=(patternSize[1],patternSize[0])
    x,y,z=np.zeros((patternSizeT)),np.zeros((patternSizeT)),np.zeros((patternSizeT))
    xo,yo,zo=np.zeros((patternSizeT)),np.zeros((patternSizeT)),np.zeros((patternSizeT))
    j=0; n=patternSize[0]
    for i in range(patternSize[1]):
        x[i,:]=X[j:j+n]; y[i,:]=Y[j:j+n]; z[i,:]=Z[j:j+n]
        xo[i,:]=objp[j:j+n,0]; yo[i,:]=objp[j:j+n,1]; zo[i,:]=objp[j:j+n,2]
        j+=n


    errX=(xo-x); errY=(yo-y); errZ=(zo-z)

    errXrms=np.sqrt((errX**2).mean())
    errYrms=np.sqrt((errY**2).mean())
    errZrms=np.sqrt((errZ**2).mean())
    errRMS=np.sqrt( (errX**2 + errY**2 + errZ**2).mean() )
    # --------------------------------------------------------------------------

    s = cv.FileStorage()
    s.open('mesures/image_{}.xml'.format(nb), cv.FileStorage_WRITE)
    s.write('x', x)
    s.write('y', y)
    s.write('z', z)
    s.write('errx', errXrms)
    s.write('erry', errYrms)
    s.write('errz', errZrms)
    s.write('errtot', errRMS)
    s.release()

    return errRMS

# Images à reconstruire:
i='20' #numero de l'image
mesures(i)
