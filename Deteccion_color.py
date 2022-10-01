#!/usr/bin/env python3


import matplotlib.pyplot as plt
import numpy as np
import ros_numpy
import rospy
import tf
from gazebo_ros import gazebo_interface
from sensor_msgs.msg import LaserScan, PointCloud2
from geometry_msgs.msg import Pose, Quaternion ,TransformStamped
import sys
import moveit_commander
import moveit_msgs.msg
from utils_notebooks import *
from tf2_sensor_msgs.tf2_sensor_msgs import do_transform_cloud

import cv2
import os



def correct_points(low_plane,high_plane):

    #Corrects point clouds "perspective" i.e. Reference frame head is changed to reference frame map
    data = rospy.wait_for_message('/hsrb/head_rgbd_sensor/depth_registered/rectified_points', PointCloud2)
    np_data=ros_numpy.numpify(data)
    trans,rot=listener.lookupTransform('/map', '/head_rgbd_sensor_gazebo_frame', rospy.Time(0)) 
    
    eu=np.asarray(tf.transformations.euler_from_quaternion(rot))
    t=TransformStamped()
    rot=tf.transformations.quaternion_from_euler(-eu[1],0,0)
    t.header.stamp = data.header.stamp
    
    t.transform.rotation.x = rot[0]
    t.transform.rotation.y = rot[1]
    t.transform.rotation.z = rot[2]
    t.transform.rotation.w = rot[3]

    cloud_out = do_transform_cloud(data, t)
    np_corrected=ros_numpy.numpify(cloud_out)
    corrected=np_corrected.reshape(np_data.shape)

    img= np.copy(corrected['y'])

    img[np.isnan(img)]=2
    #img3 = np.where((img>low)&(img< 0.99*(trans[2])),img,255)
    img3 = np.where((img>0.99*(trans[2])-high_plane)&(img<0.99*(trans[2])-low_plane),img,255)
    print(trans[2])
    return img3

def main():

    print("INITIALAZING POINT CLOUD VIEWER")
    rospy.init_node('Este_si') 
    rgbd = RGBD()
    global listener, broadcaster
    listener = tf.TransformListener()
    broadcaster = tf.TransformBroadcaster()
    aux=0
    
    
    while (aux==0):
        image=rgbd.get_image() 
        if np.any(image!= None):
            plt.imshow(rgbd.get_image())
            plt.show()
            aux=1
    
    while (aux==1):
        points=rgbd.get_points() 
        if np.any(points!= None):
            plt.imshow(points['z'])
            #plt.show()
            plt.imshow(points['z'],cmap='YlOrRd')
            plt.show()
            aux=0
    while (aux==0):
        corrected=correct_points(0.4,0.8) #Altura
        plt.imshow(corrected)
        plt.show()
        aux=2 
        
    
    contours, hierarchy = cv2.findContours(corrected.astype('uint8'),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    print(len(contours))
    
    im_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    im_hsv = cv2.cvtColor(im_bgr, cv2.COLOR_BGR2HSV)
    
    for contour in contours:
        M = cv2.moments(contour)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        boundRect = cv2.boundingRect(contour)
        image2=cv2.rectangle(im_hsv,(boundRect[0], boundRect[1]),(boundRect[0]+boundRect[2], boundRect[1]+boundRect[3]),(255,255,255),2)
        cv2.circle(image2, (cX, cY), 5, (255,255,255),-1)
        cv2.putText(image2, "centroid_"+str(cX)+','+str(cY) , (cX - 50, cY - 25) , cv2.FONT_HERSHEY_SIMPLEX, 0.9,(255,255,255),2)
        
    plt.imshow(cv2.cvtColor(image2, cv2.COLOR_HSV2RGB))
    plt.show()
    """
   ###############################Crear la mascara###################
    h_min=30
    h_max=100

    region = (im_hsv > h_min) & (im_hsv < h_max)
    idx,idy=np.where(region[:,:,0] )
    
    mask= np.zeros((480,640))
    mask[idx,idy]=255
    
    ##################################################################
    
   
   
    while (aux==2):
        if np.any(mask!= None):
            plt.imshow(mask,cmap='gray')
            plt.show()
        aux=3
    kernel = np.ones((5,5), np.uint8)
    eroded_mask=cv2.erode(mask,kernel)
    dilated_mask=cv2.dilate(eroded_mask,kernel)
    while (aux==3):
        if np.any(dilated_mask!= None):
            plt.imshow(dilated_mask,cmap='gray')
            plt.show()
        aux=4
    #plt.imshow(dilated_mask,cmap='gray')"""

if __name__ == '__main__':  
    
    main()
