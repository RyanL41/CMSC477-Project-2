"""
AprilTag detector module to handle detection and pose estimation of AprilTags.
"""
import numpy as np
import pupil_apriltags
from scipy.spatial.transform import Rotation as R
from Project3.config import SCALE_FACTOR

class AprilTagDetector:
    def __init__(self, camera_matrix, marker_size_m=0.16, family="tag36h11", threads=2):
        """
        Initialize the AprilTag detector.
        
        Args:
            camera_matrix: Camera calibration matrix
            marker_size_m: Size of the AprilTag in meters
            family: AprilTag family type
            threads: Number of threads to use for detection
        """
        self.camera_params = [
            camera_matrix[0][0], camera_matrix[1][1], 
            camera_matrix[0][2], camera_matrix[1][2]
        ]
        self.marker_size_m = marker_size_m
        self.detector = pupil_apriltags.Detector(family, threads)
        
    def find_tags(self, frame_gray):
        """
        Detect AprilTags in the grayscale image.
        
        Args:
            frame_gray: Grayscale image
            
        Returns:
            List of AprilTag detections
        """
        return self.detector.detect(
            frame_gray,
            estimate_tag_pose=True,
            camera_params=self.camera_params,
            tag_size=self.marker_size_m,
        )

    @staticmethod
    def get_pose_from_detection(detection):
        """Extract pose information from AprilTag detection"""
        r_matrix = np.array(detection.pose_R).reshape(3, 3)  # Rotation matrix
        t_vector = np.array(detection.pose_t).flatten()      # Translation vector
        return t_vector, r_matrix
        
    @staticmethod
    def get_tag_id(detection):
        """Get AprilTag ID from detection"""
        return detection.tag_id
        
    def get_tag_world_position(self, detection, robot_pos, robot_yaw):
        """
        Computes the AprilTag's position in world frame.
        
        Args:
            detection: AprilTag detection
            robot_pos: Robot position (x, y) in world frame
            robot_yaw: Robot yaw angle in degrees
            
        Returns:
            Tag position in world coordinates (x, y)
        """
        # Get tag pose in camera frame
        t_ca, r_ca = self.get_pose_from_detection(detection)

        print("TCA",t_ca)
        print("RCa",r_ca)
        
        # Compute the offset of the tag in camera frame
        offset = np.array([-t_ca[0] / SCALE_FACTOR, t_ca[2] / SCALE_FACTOR])

        print("Offset",offset)
        
        # Extract relative rotation (yaw) from the detection
        rot = R.from_matrix(r_ca)
        z_rot = rot.as_euler("xyz", degrees=False)[1]

        print("Z_rot",z_rot)
        
        # Rotate offset based on camera-to-tag orientation
        r_z_rot = np.array([
            [np.cos(-z_rot), -np.sin(-z_rot)], 
            [np.sin(-z_rot), np.cos(-z_rot)]
        ])
        offset = r_z_rot.dot(offset)
        offset[1] = -offset[1]
        
        # Create rotation matrix for robot's orientation in world frame
        robot_yaw_rad = robot_yaw * (np.pi / 180.0)  # Convert to radians
        r_robot = np.array([
            [np.cos(robot_yaw_rad), -np.sin(robot_yaw_rad)], 
            [np.sin(robot_yaw_rad), np.cos(robot_yaw_rad)]
        ])

        print("Rrobot",r_robot)
        
        # Transform the offset to world coordinates
        world_offset = r_robot.dot(offset)

        print("World offset",world_offset)        
        # Tag position = robot position - rotated offset
        tag_world_pos = np.array(robot_pos[:2]) - world_offset
        
        return tag_world_pos
