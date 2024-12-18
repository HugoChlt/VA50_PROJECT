#!/usr/bin/env python3

import rospy
import sys
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Header

def send_goal(x, y, theta):
    # Initialize the ROS node
    rospy.init_node('move_base_goal_publisher')

    # Create a publisher that will publish to move_base_simple/goal
    pub = rospy.Publisher('/robot/move_base_simple/goal', PoseStamped, queue_size=10)

    # Wait until the publisher is connected
    rospy.sleep(1)

    # Create a PoseStamped message
    goal = PoseStamped()

    # Set header information
    goal.header = Header()
    goal.header.stamp = rospy.Time.now()
    goal.header.frame_id = "robot_map"  # Coordinate frame for the goal (usually "map")

    # Set the position (X, Y) and orientation (theta) for the goal
    goal.pose.position.x = x
    goal.pose.position.y = y
    goal.pose.position.z = 0.0

    # Convert theta (yaw) to quaternion (for orientation)
    quat = euler_to_quaternion(0.0, 0.0, theta)  # No roll, no pitch, just yaw

    goal.pose.orientation.x = quat[0]
    goal.pose.orientation.y = quat[1]
    goal.pose.orientation.z = quat[2]
    goal.pose.orientation.w = quat[3]

    # Publish the goal
    rospy.loginfo("Sending goal: (%f, %f) with orientation: %f", x, y, theta)
    pub.publish(goal)

def euler_to_quaternion(roll, pitch, yaw):
    """
    Convert Euler angles to a quaternion.
    """
    from tf.transformations import quaternion_from_euler
    return quaternion_from_euler(roll, pitch, yaw)

if __name__ == '__main__':
    try:
        # Check if coordinates are provided via command-line arguments
        if len(sys.argv) != 4:
            rospy.logerr("Usage: rosrun <package_name> <script_name.py> <x> <y> <theta>")
            sys.exit(1)
        
        # Parse the coordinates from command-line arguments
        x = float(sys.argv[1])
        y = float(sys.argv[2])
        theta = float(sys.argv[3])

        # Call the function with the coordinates
        send_goal(x, y, theta)  # You can change these values as needed

    except rospy.ROSInterruptException:
        pass