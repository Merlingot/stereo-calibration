import cv2 as cv
import numpy as np

doUpdate = False
pts2d = []

def onClick(event, x, y, flags, param):
    global pts2d, doUpdate
    if event != cv.EVENT_LBUTTONDOWN:
        return
    pts2d.append([x, y])
    doUpdate = True


def drawPoints(image, points):
    img = image.copy()
    for p in points:
        center = (int(p[0]), int(p[1]))
        print("center", center)
        img = cv.circle(img, center, 10, (255), 1)
        img = cv.circle(img, center, 11, (0), 1)
    return img

def clickOnTheTargets(img_rgb):
    global doUpdate, pts2d
    img_gray = cv.cvtColor(img_rgb, cv.COLOR_BGR2GRAY)

    cv.namedWindow("image")
    cv.setMouseCallback("image", onClick)

    img_gray_doodles = img_gray.copy()

    while True:
        if doUpdate:
            print("UPDATE")
            img_gray_doodles = drawPoints(img_gray, pts2d)
            doUpdate = False
        cv.imshow("image", img_gray_doodles)
        key = cv.waitKey(1) & 0xFF

        if key == ord("q"):
            exit()
        elif key == ord("c"):
            break
        elif key == ord("z"):
            if len(pts2d) > 0:
                del pts2d[-1]
                doUpdate = True
            print("ZZZZZZZZZZZZZZZZZZ")

    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.0001)
    corners = cv.cornerSubPix(img_gray, np.float32(pts2d), (3,3), (-1,-1), criteria)
    img = img_rgb.copy()
    for p in corners:
        center = (int(p[0]), int(p[1]))
        img = cv.circle(img, center, 3, (0,0,255), 1)
    return corners, img

# def click_n_save(img_rgb):
#     corners, img = clickOnTheTargets(img_rgb)

# # LEFT
# img = cv.imread("captures_zed/cibles_40m/cibles_5MPX/right2.jpg")
# pts, img = clickOnTheTargets(img)
#
# cv.imwrite("cornerSubPix_test.png", img)
# # np.savetxt("pts_left.txt", pts)
# # with open('pts_left.npy', 'wb') as f:
#     # np.save(f, pts)
#
# # RIGHT
# # img = cv.imread("captures_zed/damier_cibles/right2.jpg")
# # pts, img = clickOnTheTargets(img)
# #
# # cv.imwrite("cornerSubPix_right.png", img)
# # np.savetxt("pts_right.txt", pts)
# # with open('pts_right.npy', 'wb') as f:
# #     np.save(f, pts)
#
# # pts = np.loadtxt("pts_left.txt")
# # print(pts)
# # print("the end")
