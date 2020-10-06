###############################run this script###########################

import numpy as np
import cv2
import glob
from IPM import ipm

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
o = np.zeros((6*7,3), np.float32)  # check chesboard pose 
for i in range(6):
    for j in range(i*7,i*7+7):
        o[j,0]=5-i
        o[j,1]=j%7
# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

images = glob.glob('chessboard/*.jpg')


for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (7,6),None)

    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(o)

        corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        imgpoints.append(corners2)

        # Draw and display the corners
        cv2.namedWindow("img",cv2.WINDOW_NORMAL)        # Create window with freedom of dimensions
        img = cv2.drawChessboardCorners(img, (7,6), corners2,ret)
        cv2.imshow('img',img)
        cv2.waitKey(500)

cv2.destroyAllWindows()

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)

for idx in range(len(images)):
    img=cv2.imread(images[idx])
    points_w=objpoints[idx]
    points_f=imgpoints[idx]
    
    intrinsic_mtx=mtx
    
    rot_vec=rvecs[idx]
    rot_mtx=cv2.Rodrigues(rot_vec)[0]
    tran_mtx=tvecs[idx]
    
    ipm_img=ipm(intrinsic_mtx,rot_mtx,tran_mtx,img)
    
    cv2.imwrite("IPM_result/Chessboard_IPM_result"+str(idx)+".png",ipm_img)
