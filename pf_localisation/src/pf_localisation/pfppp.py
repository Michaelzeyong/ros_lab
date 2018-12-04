from geometry_msgs.msg import Pose, PoseArray, Quaternion
from pf_base import PFLocaliserBase
import math
import rospy
import numpy as np
import random

from util import rotateQuaternion, getHeading
from random import random

from time import time


class PFLocaliser(PFLocaliserBase):
       
    def __init__(self):
        # Call the superclass constructor
        super(PFLocaliser, self).__init__()
        
        # Set motion model parameters
        self.ODOM_ROTATION_NOISE = 0.05 # Odometry model rotation noise
        self.ODOM_TRANSLATION_NOISE = 0.5 # Odometry model x axis (forward) noise
        self.ODOM_DRIFT_NOISE = 0.5 # Odometry model y axis (side-to-side) noise
        self.RESAMPING_NOISE = 0.005 # Noise added to resampled particle cloud
 
        # Sensor model parameters
        self.NUMBER_PREDICTED_READINGS = 20 	# Number of readings to predict
        self.NUMBER_PARTICLE = 2000 #Number of particle
        self.INDEX_MAX_WEIGHT = 0  #The max weight pose
        self.MEAN_MAX_WEIGHT_POSE_X = 0
        self.MEAN_MAX_WEIGHT_POSE_Y = 0
        self.VARIANCE_MAX_WEIGHT_POSE_X = 0
        self.VARIANCE_MAX_WEIGHT_POSE_X = 0
        self.NUMBER_HIGE_WEIGHT = 0
        self.ESTIMATE_MAX_WEIGHT = 0
       
    def initialise_particle_cloud(self, initialpose):
        # Set particle cloud to initialpose plus noise
        """
        Called whenever an initialpose message is received (to change the
        starting location of the robot), or a new occupancy_map is received.
        
        :Args:
            | initialpose: the initial pose estimate
        :Return:
            | (geometry_msgs.msg.PoseArray) poses of the particles
        """
        pose_array = PoseArray()

        for i in range(0,self.NUMBER_PARTICLE):
            temp = Pose()
            rnd = []
            for j in range(0,3):
                rnd.append(np.random.normal(0,1))

            temp.position.x = initialpose.pose.pose.position.x + self.ODOM_TRANSLATION_NOISE*rnd[0]
            temp.position.y = initialpose.pose.pose.position.y + self.ODOM_DRIFT_NOISE*rnd[1]
            temp.position.z = 0
            temp.orientation = rotateQuaternion(initialpose.pose.pose.orientation, rnd[2]*self.ODOM_ROTATION_NOISE*2*math.pi)
            pose_array.poses.extend([temp])
        return pose_array
    
    def update_particle_cloud(self, scan):
        # Update particlecloud, given map and laser scan
        # through Zt -> prior bel -> posterior bel  
        """
        This should use the supplied laser scan to update the current
        particle cloud. I.e. self.particlecloud should be updated.
        
        :Args:
            | scan (sensor_msgs.msg.LaserScan): laser scan to use for update

         """
        resample_particlecloud = PoseArray()
        weight_particle = []
        cumulative_weight_particle = []
        high_weight_number = 0
        for i in range(0,self.NUMBER_PARTICLE):
            #for get weight
            weight_temp = self.sensor_model.get_weight(scan,self.particlecloud.poses[i])
            weight_particle.append(weight_temp)
            #resample delete low weight particle
        
        sum_weight = sum(weight_particle)
        for i in xrange(0,len(weight_particle)):
            cumulative_weight_particle.extend([weight_particle[i]/sum_weight])

        for i in xrange(0,self.NUMBER_PARTICLE):
            if cumulative_weight_particle[i] > 1/self.NUMBER_PARTICLE:
                resample_particlecloud.poses.extend([self.particlecloud.poses[i]])
                high_weight_number = high_weight_number+1


        # max_weight = np.where(cumulative_weight_particle == np.argmax(cumulative_weight_particle))
        # index_max_weight = max_weight[0][0]
        index_max_weight = np.argmax(cumulative_weight_particle)
        sum_pose_x =0
        sum_pose_y =0
        # sum_pose_x = sum(resample_particlecloud.poses.position.x)
        for i in range(0,high_weight_number):
            sum_pose_x = sum_pose_x + resample_particlecloud.poses[i].position.x
        for i in range(0,high_weight_number):
            sum_pose_y += resample_particlecloud.poses[i].position.y
        
        mean_x = sum_pose_x/high_weight_number
        mean_y = sum_pose_y/high_weight_number

        variance_x = abs(mean_x - self.particlecloud.poses[index_max_weight].position.x) / 8
        variance_y = abs(mean_y - self.particlecloud.poses[index_max_weight].position.y) / 8
        #generate new particles 
        for num_particle in range(0,self.NUMBER_PARTICLE - high_weight_number):
            temp_pose = self.particlecloud.poses[index_max_weight]
            rnd_position_x = random.gauss(temp_pose.poses.position.x , variance_x)
            rnd_position_y = random.gauss(temp_pose.poses.position.y , variance_y)
            #noise
            rnd = []
            for j in range(0,3):
                rnd.append(np.random.normal(0,1))
            temp_pose.position.x = rnd_position_x + rnd[0]*self.RESAMPING_NOISE
            temp_pose.position.y = rnd_position_y + rnd[1]*self.RESAMPING_NOISE
            temp_pose.orientation = rotateQuaternion(temp_pose.orientation, rnd[2]*0.001*2*math.pi)
            resample_particlecloud.poses.extend([temp_pose])
        #transfer the max index of particle
        self.ESTIMATE_MAX_WEIGHT = self.particlecloud.poses[index_max_weight]

        #return the particlecloud
        self.particlecloud = resample_particlecloud       
        #transfer parameter to 
        self.INDEX_MAX_WEIGHT = index_max_weight
        self.MEAN_MAX_WEIGHT_POSE_X = mean_x
        self.MEAN_MAX_WEIGHT_POSE_Y = mean_y
        self.VARIANCE_MAX_WEIGHT_POSE_X = variance_x
        self.VARIANCE_MAX_WEIGHT_POSE_X = variance_y
        self.NUMBER_HIGE_WEIGHT = high_weight_number
        '''#get 500 particle that have high w
        for m in xrange(0,500):
            max_weight=np.where(weight_particle == np.max(weight_particle))
                for i in xrange(len(max_weight[0])):
                    temp = max_weight[0][i] - i
                    particlecloud_temp.poses.append(particlecloud.poses[temp])
                    del weight_particle[temp]`
                m = m + len(max_weight[0])
        '''

        




    def estimate_pose(self):
        # Create new estimated pose, given particle cloud
        # E.g. just average the location and orientation values of each of
        # the particles and return this.
        
        # Better approximations could be made by doing some simple clustering,
        # e.g. taking the average location of half the particles after 
        # throwing away any which are outliers
        """
        This should calculate and return an updated robot pose estimate based
        on the particle cloud (self.particlecloud).
        
        :Return:
            | (geometry_msgs.msg.Pose) robot's estimated pose.
         """
        
        #function 1 : weight_resample particlecloud
        pose = Pose()
        
        for i in xrange(0,self.NUMBER_PARTICLE - self.NUMBER_HIGE_WEIGHT):
            temp_pose = self.ESTIMATE_MAX_WEIGHT.poses
            temp_pose.position.x = np.random.normal(temp_pose.position.x , self.VARIANCE_MAX_WEIGHT_POSE_X)
            temp_pose.position.y = np.random.normal(temp_pose.position.y , self.VARIANCE_MAX_WEIGHT_POSE_Y)
            temp_pose.orientation = temp_pose.orientation
            pose.extend([temp_pose])
        #rospy.loginfo(pose)
        return pose

        #function 2 :max_weight 
        # pose = self.ESTIMATE_MAX_WEIGHT.poses
        # return pose

        #function3 : Average positon
        # pose = self.ESTIMATE_MAX_WEIGHT.poses
        # mean_x = np.average(self.particlecloud.poses.position.x)
        # mean_y = np.average(self.particlecloud.poses.position.y)
        # pose.position.x = mean_x
        # pose.position.y = mean_y
        # return pose 
