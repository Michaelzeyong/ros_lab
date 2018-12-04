#!/usr/bin/env python
import rospy
import actionlib
from actionlib_msgs.msg import *
from geometry_msgs.msg import Pose, Point, Quaternion, Twist, PoseWithCovarianceStamped
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from tf.transformations import quaternion_from_euler
from visualization_msgs.msg import Marker
from math import radians, pi
from face_knn.knnclass import *

from nav_msgs.msg import OccupancyGrid, Odometry
import pyttsx

class Final:
    def __init__(self):
        rospy.init_node('final', anonymous=False)

        #face_recognition
        find_guest = face_classification()

        #Define a marker publisher
        self.marker_pub = rospy.Publisher('goal_markers', Marker, queue_size=5)

        #initialize the marker points list.
        self.markers = Marker()
        
        self.init_markers()

        waypoints = list()

        waypoints.append(Pose(Point(10.86, 15.75, 0.0), Quaternion(0.0 , 0.0 , 0.12, 0.99)))
        waypoints.append(Pose(Point(16.9, 25.1, 0.0), Quaternion(0.0 , 0.0 , 0.99, -0.14)))
        waypoints.append(Pose(Point(15, 31, 0.0), Quaternion(0.0 , 0.0 , 0.99, -0.12)))
     
        # Initialize the visualization markers for RViz
        
        
        # Set a visualization marker at each waypoint        
        for waypoint in waypoints:           
            p = Point()
            p = waypoint.position
            self.markers.points.append(p)
        
        self.move_base = actionlib.SimpleActionClient("move_base", MoveBaseAction)
        
        rospy.loginfo("Waiting for move_base action server...")
        
        # Wait 60 seconds for the action server to become available
        self.move_base.wait_for_server(rospy.Duration(600))
        
        rospy.loginfo("Connected to move base server")
        rospy.loginfo("Starting navigation test")

        while True:
            self.marker_pub.publish(self.markers)
            # Intialize the waypoint goal
            goal = MoveBaseGoal()
            
            # Use the map frame to define goal poses
            goal.target_pose.header.frame_id = 'map'
            
            # Set the time stamp to "now"
            goal.target_pose.header.stamp = rospy.Time.now()

            #speak
            self.robot_spaeker('Please input the table')

            table = raw_input('Please input the table: ')
            guest = False
            if table == "q":
                break
            if int(table)==1:
                goal.target_pose.pose = waypoints[int(table)]
                result_move = self.move(goal)
                if result_move:
                    guest = find_guest.video_predict("zhangzeyong",True)
                if guest:
                    self.robot_spaeker('Please take you dishs, enjoy it!')
                    print 'Please take you dishs, enjoy it!'
                time.sleep(3)
            elif int(table)==2:
                goal.target_pose.pose = waypoints[int(table)]
                result_move = self.move(goal)
                if result_move:
                    guest = find_guest.video_predict("zhangzeyong",False)
                self.robot_spaeker('Please take you dishs, enjoy it!')
                print 'Please take you dishs, enjoy it!'
                time.sleep(3)
            else:
                print('input wrong table!!!!!')
            print 'Go back to kitchen'
            goal.target_pose.pose = waypoints[0]
            if(self.move(goal)):
                continue

    def move(self, goal):
            # Send the goal pose to the MoveBaseAction server
            self.move_base.send_goal(goal)
            
            # Allow 1 minute to get there
            finished_within_time = self.move_base.wait_for_result(rospy.Duration(600)) 
            
            # If we don't get there in time, abort the goal
            if not finished_within_time:
                self.move_base.cancel_goal()
                rospy.loginfo("Timed out achieving goal")
                return False
            else:
                # We made it!
                state = self.move_base.get_state()
                if state == GoalStatus.SUCCEEDED:
                    rospy.loginfo("Goal succeeded!")
                    return True 

    def init_markers(self):
        # Set up our waypoint markers
        marker_scale = 2
        marker_lifetime = 0 # 0 is forever
        marker_ns = 'waypoints'
        marker_id = 0
        marker_color = {'r': 1.0, 'g': 0.7, 'b': 1.0, 'a': 1.0}
        
        # Define a marker publisher.
        # self.marker_pub = rospy.Publisher('waypoint_markers', Marker, queue_size=5)
        
        # # Initialize the marker points list.
        # self.markers = Marker()
        self.markers.ns = marker_ns
        self.markers.id = marker_id
        self.markers.type = Marker.CUBE_LIST
        self.markers.action = Marker.ADD
        self.markers.lifetime = rospy.Duration(marker_lifetime)
        self.markers.scale.x = marker_scale
        self.markers.scale.y = marker_scale
        self.markers.color.r = marker_color['r']
        self.markers.color.g = marker_color['g']
        self.markers.color.b = marker_color['b']
        self.markers.color.a = marker_color['a']
        
        self.markers.header.frame_id = "/map"
        self.markers.header.stamp = rospy.Time.now()
        self.markers.points = list()

    def robot_spaeker(self,speak_words):
        #voice
        robot_spaker = pyttsx.init()
        rate = robot_spaker.getProperty('rate')
        robot_spaker.setProperty('rate', rate-40)
        robot_spaker.say(speak_words)
        robot_spaker.runAndWait()


if __name__ == '__main__':
    try:
        Final()
    except rospy.ROSInterruptException:
        rospy.loginfo("Navigation test finished.")