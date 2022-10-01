#!/usr/bin/env python3

#--------------------------------- Librerias ------------------------------------------

import rospy
import numpy as np
import time
import tf2_ros
import tf
import math
import smach
import ros_numpy
import matplotlib.pyplot as plt
import matplotlib
from gazebo_ros import gazebo_interface

from nav_msgs.msg import Odometry
from tf.transformations import euler_from_quaternion
from math import atan2
from geometry_msgs.msg import PoseStamped,Pose
from utils_evasion import *
from sensor_msgs.msg   import LaserScan, PointCloud2
from geometry_msgs.msg import Twist, Point, Pose, Quaternion, TransformStamped
import sys
import moveit_commander
import moveit_msgs.msg
from utils_notebooks import *
from tf2_sensor_msgs.tf2_sensor_msgs import do_transform_cloud
import cv2
import os


#------------------------------ Inicializando Nodo ------------------------------------

rospy.init_node("nodo1")
EQUIPO="UPS Cuenca"
r=rospy.Rate(4)
twist=Twist()
global or_p_x, or_p_y, Theta_goal

or_p_x=[ 0   , -3.0 , 3.9 ]
or_p_y=[1.21 ,  4.0 , 5.6]
def init(node_name):
    global laser, base_vel_pub
    base_vel_pub=rospy.Publisher('/hsrb/command_velocity', Twist, queue_size=10) 
    laser=Laser()

def gaze_point(x,y,z):
    
    
    
    head_pose = head.get_current_joint_values()
    head_pose[0]=0.0
    head_pose[1]=0.0
    head.set_joint_value_target(head_pose)
    head.go()
    
    trans , rot = listener.lookupTransform('/map', '/head_rgbd_sensor_gazebo_frame', rospy.Time(0)) #
    
 
    
    e =tf.transformations.euler_from_quaternion(rot)
    

    x_rob,y_rob,z_rob,th_rob= trans[0], trans[1] ,trans[2] ,  e[2]


    D_x=x_rob-x
    D_y=y_rob-y
    D_z=z_rob-z

    D_th= np.arctan2(D_y,D_x)
    print('relative to robot',(D_x,D_y,np.rad2deg(D_th)))

    pan_correct= (- th_rob + D_th + np.pi) % (2*np.pi)

    if(pan_correct > np.pi):
        pan_correct=-2*np.pi+pan_correct
    if(pan_correct < -np.pi):
        pan_correct=2*np.pi+pan_correct

    if ((pan_correct) > .5 * np.pi):
        print ('Exorcist alert')
        pan_correct=.5*np.pi
    head_pose[0]=pan_correct
    tilt_correct=np.arctan2(D_z,np.linalg.norm((D_x,D_y)))

    head_pose [1]=-tilt_correct
    
    
    
    head.set_joint_value_target(head_pose)
    succ=head.go()
    return succ
    
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
    return img3

def Obtener_datos_camara(flag):
    #while (flag==True):
    image=rgbd.get_image() 
    """if np.any(image!= None):
            plt.imshow(rgbd.get_image())
            plt.show()
            flag = False
    """    
    points=rgbd.get_points() 
    corrected=correct_points(0,2) #Altura
    contours, hierarchy = cv2.findContours(corrected.astype('uint8'),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    print("Contornos Detectados: "+str(len(contours)))
    
    im_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)	
    im_hsv = cv2.cvtColor(im_bgr, cv2.COLOR_BGR2HSV)
    
    for contour in contours:
        xyz=[]
        M = cv2.moments(contour)
        
        print(M)
        
        if int(M["m00"])>99 and int(M["m00"])<9999 and int(M["m00"])>0 and int(M["m10"])>9999 and int(M["m10"])<9999999:
            #print(M["m00"])
            #print(M["m10"])
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            boundRect = cv2.boundingRect(contour)
            image2=cv2.rectangle(im_hsv,(boundRect[0], boundRect[1]),(boundRect[0]+boundRect[2], boundRect[1]+boundRect[3]),(255,255,255),2)
            cv2.circle(image2, (cX, cY), 5, (255,255,255),-1)
            cv2.putText(image2, "centroid_"+str(cX)+','+str(cY) , (cX - 50, cY - 25) , cv2.FONT_HERSHEY_SIMPLEX, 0.9,(255,255,255),2)
    for contour in contours:
        xyz=[]
        M = cv2.moments(contour)
        
        if int(M["m00"])>0:
        
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            boundRect = cv2.boundingRect(contour)
            for jy in range (boundRect[0], boundRect[0]+boundRect[2]):
                for ix in range(boundRect[1], boundRect[1]+boundRect[3]):
                    aux=(np.asarray((points['x'][ix,jy],points['y'][ix,jy],points['z'][ix,jy])))
                    if np.isnan(aux[0]) or np.isnan(aux[1]) or np.isnan(aux[2]):
                        'reject point'
                    else:
                        xyz.append(aux)

            xyz=np.asarray(xyz)
            cent=xyz.mean(axis=0)
            print(cent)
            x,y,z=cent
            print(x)
            if np.isnan(x) or np.isnan(y) or np.isnan(z):
                print('nan')
            else:
                
                broadcaster.sendTransform((x,y,z),(0,0,0,1), rospy.Time.now(), 'Object',"head_rgbd_sensor_link")
                ##    xm=listener.lookupTransform('map','Object',rospy.Time(0))
         
            broadcaster.sendTransform((0,0,0),(0,0,0,1), rospy.Time.now(), 'Object_fix','map')
    plt.imshow(cv2.cvtColor(image2, cv2.COLOR_HSV2RGB))
    plt.show()


#--------------------------------- Puntos Objetos -------------------------------------
#aumentar estados a la variable actual_points
def points(actual_point):
# primer centroide
    if actual_point==1:
    	
        x_goal,y_goal=0, 0.11 #0,1.21-1.1
        Theta_goal=np.pi/2 #atan2(or_p_y[0], or_p_x[0])
        return x_goal, y_goal,Theta_goal
        

    elif actual_point==2:
        x_goal, y_goal=-1.4, 1.21 #-1.1, 1.21
        Theta_goal=0
        #Theta_goal=atan2(or_p_y[0], or_p_x[0])
        return x_goal, y_goal,Theta_goal
        

#Segundo centroide
    elif actual_point==3:
        x_goal, y_goal = -3, 2.2  #-3.0, 4.0-1.1
        Theta_goal=np.pi/2 #atan2(or_p_y[1], or_p_x[1])
        return x_goal, y_goal,Theta_goal

    elif actual_point==4:
        x_goal, y_goal = -1.1, 4.0
        Theta_goal=np.pi #atan2(or_p_y[1], or_p_x[1])
        return x_goal, y_goal,Theta_goal
  
#Teercer centroide
    elif actual_point==5:
        x_goal,y_goal= 2, 5.4#3.9-1.1, 5.6
        Theta_goal=0 #atan2(or_p_y[2], or_p_x[2])
        return x_goal,y_goal,Theta_goal

    elif actual_point==6:
        x_goal,y_goal= 3.9, 6.7#, 5.6-1.1
        Theta_goal=-np.pi/2 #atan2(or_p_y[2], or_p_x[2])
        return x_goal,y_goal,Theta_goal
    
#------------------------- Funcion Acondicionamiento Laser ---------------------------
        
def get_lectura_cuant():
    try:
        lectura=np.asarray(laser.get_data().ranges)
        lectura=np.where(lectura>20,20,lectura) #remove infinito
        right_scan=lectura[:180]
        left_scan=lectura[540:]
        front_scan=lectura[180:540]
        sd,si,sf=0,0,0
        if np.mean(left_scan)< 1.5: si=1
        if np.mean(right_scan)< 1.5: sd=1
        if np.mean(front_scan)< 1.8: sf=1
    except:
        sd,si,sf=0,0,0    
    return si,sd,sf

#---------------------------- Funcion Coordenadas ------------------------------------

def get_coords ():
    for i in range(10):   ###TF might be late, try 10 times
        try:
            trans=tfBuffer.lookup_transform('map', 'base_link', rospy.Time())
            return trans     
        except:
            trans=0    

#------------------------- Funciones Movimiento Robot --------------------------------

def move_base_vel(vx, vy, vw):
    twist.linear.x=vx
    twist.linear.y=vy
    twist.angular.z=vw 
    base_vel_pub.publish(twist)
def move_base(x,y,yaw,timeout):
    start_time=rospy.Time.now().to_sec()
    while rospy.Time.now().to_sec()-start_time<timeout:  
        move_base_vel(x, y, yaw) 

#---------------------------------- Estados --------------------------------------    

class S0(smach.State):
    def __init__(self):
        smach.State.__init__(self, outcomes=['outcome1','outcome2','outcome3','outcome4','outcome5','outcome6','outcome7','outcome8','outcome9'])
        self.counter=0
    def execute(self,userdata):
        global theta_Goal,angle_to_goal, angle_to_goal1, xt, yt, inc_x, inc_y, theta
        x_goal,y_goal,theta_Goal=points(next_point)
        print('Punto ' + str(next_point))
        si,sd,sf=get_lectura_cuant()
        msg_cmd_vel=Twist()
        punto_act=get_coords()
        xt=punto_act.transform.translation.x
        yt=punto_act.transform.translation.y
        (roll, pitch, theta) = euler_from_quaternion ([punto_act.transform.rotation.x, punto_act.transform.rotation.y, punto_act.transform.rotation.z, punto_act.transform.rotation.w])	
        inc_x=x_goal-xt
        inc_y=y_goal-yt
        angle_to_goal=atan2(inc_y, inc_x)
        angle_to_goal1=angle_to_goal-theta
              	
        	 
        	
        if (si==0 and sd==0 and sf==0): return 'outcome9'     
        if (si==0 and sf==0 and sd==1): return 'outcome2'
        if (si==0 and sf==1 and sd==0): return 'outcome3'
        if (si==0 and sf==1 and sd==1): return 'outcome4'
        if (si==1 and sf==0 and sd==0): return 'outcome5'
        if (si==1 and sf==0 and sd==1): return 'outcome6'
        if (si==1 and sf==1 and sd==0): return 'outcome7'
        if (si==1 and sf==1 and sd==1): return 'outcome8'
        return 'outcome1' 
        pub_cmd_vel.publish(msg_cmd_vel)
        r.sleep() 

class S1(smach.State):
    def __init__(self):
        smach.State.__init__(self, outcomes=['outcome1','outcome2'])
        self.counter=0
    def execute(self,userdata):
        rospy.loginfo('Estado Derecha')
        if abs(inc_x)>0.05 and abs(inc_y)>0.05:
            move_base(0.3,0.0,0.12*np.pi,0.08)
        return 'outcome1'

class S2(smach.State):
    def __init__(self):
        smach.State.__init__(self, outcomes=['outcome1','outcome2'])
        self.counter=0
    def execute(self,userdata):
        rospy.loginfo('Estado Frente')
        if abs(inc_x)>0.05 and abs(inc_y)>0.05:
            if ((angle_to_goal1>=0 and angle_to_goal1<np.pi) or (angle_to_goal1<-np.pi)):
                if abs(angle_to_goal1)>0.1: move_base(0.0,0.0,0.12*np.pi,0.1)                 
            elif ((angle_to_goal1>=np.pi and angle_to_goal1<2*np.pi) or (angle_to_goal1<0 and angle_to_goal1>-(np.pi))):
                if abs(angle_to_goal1)>0.1: move_base(0.0,0.0,-0.12*np.pi,0.1)  
        return 'outcome1'

class S3(smach.State):
    def __init__(self):
        smach.State.__init__(self, outcomes=['outcome1','outcome2'])
        self.counter=0
    def execute(self,userdata):
        rospy.loginfo('Estado Frente - Derecha')
        if abs(inc_x)>0.05 and abs(inc_y)>0.05:
            move_base(0.0,0.0,0.12*np.pi,0.1)
        return 'outcome1'

class S4(smach.State):
    def __init__(self):
        smach.State.__init__(self, outcomes=['outcome1','outcome2'])
        self.counter=0
    def execute(self,userdata):
        rospy.loginfo('Estado Izquierda')
        if abs(inc_x)>0.05 and abs(inc_y)>0.05:
            move_base(0.3,0.0,-0.12*np.pi,0.08)
        return 'outcome1' 

class S5(smach.State):
    def __init__(self):
        smach.State.__init__(self, outcomes=['outcome1','outcome2'])
        self.counter=0
    def execute(self,userdata):
        rospy.loginfo('Estado Izquierda - Derecha')
        move_base(0.3,0,0,0.1)
        return 'outcome1' 

class S6(smach.State):
    def __init__(self):
        smach.State.__init__(self, outcomes=['outcome1','outcome2'])
        self.counter=0
    def execute(self,userdata):
        rospy.loginfo('Estado Frente - Izquierda')
        if abs(inc_x)>0.05 and abs(inc_y)>0.05:
            move_base(0.0,0.0,-0.12*np.pi,0.1)
        return 'outcome1' 

class S7(smach.State):
    def __init__(self):
        smach.State.__init__(self, outcomes=['outcome1','outcome2'])
        self.counter=0
    def execute(self,userdata):
        rospy.loginfo('Estado Reversa')
        move_base(-0.3,0,0,0.1)
        return 'outcome1' 

class S8(smach.State):
    def __init__(self):
        smach.State.__init__(self, outcomes=['outcome1','outcome2'])
        self.counter=0
    def execute(self,userdata):
        rospy.loginfo('Estado Comparacion')
        if abs(inc_x)<0.05 and abs(inc_y)<0.05: 
            global angle_to_goal1 
            angle_to_goal_p=theta_Goal-theta 
            if ((angle_to_goal_p >= 0 and angle_to_goal_p <np.pi) or (angle_to_goal_p <-np.pi)):
                if abs(angle_to_goal1)>0.1: move_base(0.0,0.0,0.12*np.pi,0.1)  #move_base(0.0,0.0,theta_Goal,0.1)     
            elif ((angle_to_goal_p >=np.pi and angle_to_goal_p <2*np.pi) or (angle_to_goal_p < 0 and angle_to_goal_p > -(np.pi) )):
                if abs(angle_to_goal1)>0.1: move_base(0.0,0.0,-0.12*np.pi,0.1)  #move_base(0.0,0.0,-theta_Goal,0.1) # 
            
            if abs(angle_to_goal_p) <0.1:
                punto_fin=get_coords()
                (roll_final, pitch_final, theta_final) = euler_from_quaternion ([punto_fin.transform.rotation.x, punto_fin.transform.rotation.y, punto_fin.transform.rotation.z, punto_fin.transform.rotation.w])
                print('Tiempo = '+ str(punto_fin.header.stamp.to_sec()))
                print("x final: " + str(punto_fin.transform.translation.x))
                print("y final: " + str(punto_fin.transform.translation.y))
                print("Theta Final: " + str(theta_final))
                print("Buscando")
                global next_point
                if next_point==1:
                    head.go(np.array((0,-.15*np.pi)))
                    time.sleep(5)
                if next_point==2:
                    head.go(np.array((0,-.15*np.pi)))
                    time.sleep(5)
                if next_point==3:
                    head.go(np.array((0,-.18*np.pi)))
                    time.sleep(5)
                if next_point==4:
                    head.go(np.array((0,-.18*np.pi)))
                    time.sleep(5)
                if next_point==5:
                    head.go(np.array((0,-.18*np.pi)))
                    time.sleep(5)
                if next_point==6:
                    head.go(np.array((0,-.15*np.pi)))
                    time.sleep(5)
                
                
                Obtener_datos_camara(True)
               
                next_point=next_point+1
                if next_point==7: #4
                    while(1):
                        move_base(0,0,0,1)
        else:
            if ((angle_to_goal1>=0 and angle_to_goal1<np.pi) or (angle_to_goal1<-np.pi)):
                if abs(angle_to_goal1)>0.1: move_base(0.0,0.0,0.12*np.pi,0.1)                 
                if abs(angle_to_goal1)<0.1: move_base(0.3,0,0,0.1)
            elif ((angle_to_goal1>=np.pi and angle_to_goal1<2*np.pi) or (angle_to_goal1<0 and angle_to_goal1>-(np.pi))):
                if abs(angle_to_goal1)>0.1: move_base(0.0,0.0,-0.12*np.pi,0.1)  
                if abs(angle_to_goal1)<0.1: move_base(0.3,0,0,0.1)
        return 'outcome1'

def main():
    global pub_cmd_vel,next_point,listener, broadcaster,rgbd,head
    print("Meta Competencia Etapa 3 - " + EQUIPO)
    next_point=1	
    pub_cmd_vel=rospy.Publisher("/hsrb/command_velocity", Twist, queue_size=10)
    loop = rospy.Rate(10)
    print('Inicializando')
    rospy.sleep(1)
    punto_ini=get_coords()
    print ('Tiempo Inicial = '+ str(punto_ini.header.stamp.to_sec()))
    rgbd = RGBD()
    listener = tf.TransformListener()
    broadcaster = tf.TransformBroadcaster()
    head = moveit_commander.MoveGroupCommander('head')
    #whole_body=moveit_commander.MoveGroupCommander('whole_body_weighted')
    arm =  moveit_commander.MoveGroupCommander('arm')
    arm.set_named_target('go')
    arm.go()
    
    
    
if __name__ == '__main__':
    init("takeshi_smach")
    sm=smach.StateMachine(outcomes=['END'])     #State machine, final state "END"
    sm.userdata.sm_counter=0
    sm.userdata.clear=False   
    with sm:
        smach.StateMachine.add("s_0",   S0(),  transitions = {'outcome1':'s_0', 'outcome2':'s_1','outcome3':'s_2','outcome4':'s_3','outcome5':'s_4', 'outcome6':'s_5','outcome7':'s_6','outcome8':'s_7','outcome9':'s_8',})
        smach.StateMachine.add("s_1",   S1(),  transitions = {'outcome1':'s_8','outcome2':'END'})
        smach.StateMachine.add("s_2",   S2(),  transitions = {'outcome1':'s_8','outcome2':'END'})
        smach.StateMachine.add("s_3",   S3(),  transitions = {'outcome1':'s_8','outcome2':'END'})
        smach.StateMachine.add("s_4",   S4(),  transitions = {'outcome1':'s_8','outcome2':'END'})
        smach.StateMachine.add("s_5",   S5(),  transitions = {'outcome1':'s_8','outcome2':'END'})
        smach.StateMachine.add("s_6",   S6(),  transitions = {'outcome1':'s_8','outcome2':'END'})
        smach.StateMachine.add("s_7",   S7(),  transitions = {'outcome1':'s_2','outcome2':'END'})
        smach.StateMachine.add("s_8",   S8(),  transitions = {'outcome1':'s_0','outcome2':'END'})
    try:
        tfBuffer=tf2_ros.Buffer()
        listener=tf2_ros.TransformListener(tfBuffer)
        main()
    except rospy.ROSInterruptException:
        pass
        
outcome=sm.execute()
    
