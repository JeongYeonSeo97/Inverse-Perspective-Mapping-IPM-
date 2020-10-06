import numpy as np
import cv2

def ipm(intrinsic_mtx,rot_mtx,trans_mtx,img):
    h=img.shape[1]-1
    w=img.shape[0]-1
    
    normal_c=np.dot(rot_mtx,np.array([0,0,1]).reshape(3,1))
    origin_c=np.dot(rot_mtx,np.array([0,0,0]).reshape(3,1))+trans_mtx
    const_c=np.dot(normal_c.T,origin_c)

    corners=np.hstack((np.array([0,0,1]).reshape(3,1),
                       np.array([w,0,1]).reshape(3,1),
                       np.array([0,h,1]).reshape(3,1),
                       np.array([w,h,1]).reshape(3,1)))
    
    intrin_inv=np.linalg.inv(intrinsic_mtx)
    norm=np.dot(intrin_inv,corners)
    
    z=const_c/np.dot(normal_c.T,norm)
    
    point_c=norm*z
    
    R_inv=np.linalg.inv(rot_mtx)
    
    point_w=np.dot(R_inv,point_c-trans_mtx)
    
    xmin=np.min(point_w[0])
    xmax=np.max(point_w[0])
    
    ymin=np.min(point_w[1])
    ymax=np.max(point_w[1])
    
    point_w=point_w[:2]-np.array([xmin,ymin]).reshape(2,1)
    point_w=point_w*np.array([w,h]).reshape(2,1)/np.array([xmax-xmin,ymax-ymin]).reshape(2,1)
    
    
    
    new_pixel=np.float32(point_w.T)
    
    temp_corners=np.float32(corners[:2,:].T)
    
    Map=cv2.getPerspectiveTransform(temp_corners,new_pixel)
    
    result=cv2.warpPerspective(img,Map,(w+1,h+1))
    
    return result