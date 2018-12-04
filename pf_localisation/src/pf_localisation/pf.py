#-------------- Imports -------------#

from geometry_msgs.msg import Pose, PoseArray, Quaternion
from pf_base import PFLocaliserBase
import math
import numpy.ma as ma
import numpy as np
import rospy
from util import rotateQuaternion, getHeading
import random
from time import time


#-------------- PFLocaliser Class Methods-------------#

class PFLocaliser(PFLocaliserBase):

    #constructor
    def __init__(self):

        #Call the superclass constructor
        super(PFLocaliser, self).__init__()

        #Set motion model parameters
        self.ODOM_ROTATION_NOISE = 0.005
        self.ODOM_TRANSLATION_NOISE = 0.001
        self.ODOM_DRIFT_NOISE = 0.001

        #Sensor model readings
        self.NUMBER_PARTICLE = 1000        
    
    def initialise_particle_cloud(self, initialpose):
        #Set particle cloud to initialpose plus noise
        #create an array of Poses
        posArray = PoseArray()

        #iterate over the number of particles and append to PosArray
        for i in range(0, self.NUMBER_PARTICLE): 
            p = Pose()
            varience = 1
            p.position.x =  random.gauss(initialpose.pose.pose.position.x, varience)
            p.position.y = random.gauss(initialpose.pose.pose.position.y, varience)
            p.position.z = 0

            p.orientation = rotateQuaternion(initialpose.pose.pose.orientation, math.radians(random.gauss(0,15)))

            posArray.poses.extend([p])
            
        #print posArray
        return posArray

    def update_particle_cloud(self, scan):
        
        #list of poses
        sum_weights = 0
        weith_particle = []
        cumulative_weights_list = [] 
        normalization_list = []
        sum_count = 0

        particle_cloud = self.particlecloud.poses

        #print scan
        # print len(scan.ranges)
        scan.ranges=ma.masked_invalid(scan.ranges).filled(scan.range_max)
        # print len(scan.ranges)
        
        max_weight = 0.0
        for particle in particle_cloud:
            particle_weight = self.sensor_model.get_weight(scan, particle)
            sum_weights+= particle_weight
            weith_particle.extend([particle_weight]) 
            if particle_weight>max_weight:
            	#if weight too low replace the particles
            	max_weight = particle_weight

        for weight in weith_particle:
            weight_over_sum = weight/sum_weights
            normalization_list.extend([weight_over_sum])
            sum_count+= weight_over_sum
            cumulative_weights_list.extend([sum_count])  
        # i =0
        # sort_weight = sorted(weith_particle)
        # for weight in sort_weight:
        # 	i+=1
        # 	print("weight %d : %.4f" %(i,weight))

        
        #resample  partical cloud
        resample_particl_cloud = PoseArray()
        for particle in particle_cloud:
            rand = random.uniform(0,1)
            segment_count = 0 
            found = False    
            output = 0
            for x in range(0,len(cumulative_weights_list)):
                #if x is greater than rand
                if rand <= cumulative_weights_list[x]:
                    output = segment_count
                    found = True
                if found:
                    #print particle_cloud[output]
                    break
                segment_count+=1
            resample_particl_cloud.poses.extend([particle_cloud[output]])




        final_cloud = PoseArray()
        #add noidse
        for particle in resample_particl_cloud.poses:
            final_pose = Pose()
            final_pose.position.x = random.gauss(particle.position.x,(particle.position.x * self.ODOM_DRIFT_NOISE))
            final_pose.position.y = random.gauss(particle.position.y,(particle.position.y * self.ODOM_TRANSLATION_NOISE))
            final_pose.orientation = rotateQuaternion(particle.orientation,math.radians(random.gauss(0,15))) 

            final_cloud.poses.extend([final_pose])

        if max_weight<5:
        	self.particlecloud = self.replace_particle()
        else:
        	self.particlecloud = final_cloud

        self.weight_pose = normalization_list


    

    def estimate_pose(self):
        # 'Method to estimate the pose'
        # pose = Pose()
        # index = np.argmax(self.weight_pose)
        # pose = self.particlecloud.poses[index]
        # print(pose)

        # return pose
        x,y,z,orix,oriy,oriz,oriw,count = 0,0,0,0,0,0,0,0
        
        #iterate over each particle extracting the relevant
        #averages
        for particle in self.particlecloud.poses:
            x += particle.position.x
            y += particle.position.y
            z += particle.position.z
            orix += particle.orientation.x
            oriy += particle.orientation.y
            oriz += particle.orientation.z
            oriw += particle.orientation.w
            
        count = len(self.particlecloud.poses)

        #create a new pose with the averages of the location and 
        #orientation values of the particles
        pose = Pose()

        pose.position.x = x/count
        pose.position.y = y/count
        pose.position.z = z/count

        pose.orientation.x = orix/count
        pose.orientation.y = oriy/count
        pose.orientation.z = oriz/count
        pose.orientation.w = oriw/count

        print(pose)
        return pose




    def replace_particle(self):
    	particle_cloud = self.particlecloud.poses
    	replace_cloud = PoseArray()
        #add noidse
        for particle in particle_cloud:
            final_pose = Pose()
            final_pose.position.x = random.random()*30
            final_pose.position.y = random.random()*30
            final_pose.orientation = rotateQuaternion(Quaternion(w=1.0),math.radians(random.gauss(0,180))) 

            replace_cloud.poses.extend([final_pose])
        
        return replace_cloud

