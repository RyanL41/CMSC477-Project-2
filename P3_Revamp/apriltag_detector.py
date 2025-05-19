"""
AprilTag detector module to handle detection and pose estimation of AprilTags.
"""
import numpy as np
import pupil_apriltags
from scipy.spatial.transform import Rotation as R
from .config import SCALE_FACTOR, APRILTAG_FAMILY, APRILTAG_THREADS

class AprilTagDetector:
    def __init__(self, camera_matrix, marker_size_m=0.16, family=APRILTAG_FAMILY, threads=APRILTAG_THREADS):
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
            List of AprilTag detections or empty list if no detections or error
        """
        try:
            return self.detector.detect(
                frame_gray,
                estimate_tag_pose=True,
                camera_params=self.camera_params,
                tag_size=self.marker_size_m,
            )
        except Exception as e:
            print(f"Error detecting AprilTags: {e}")
            return []

    @staticmethod
    def get_pose_from_detection(detection):
        """
        Extract pose information from AprilTag detection.
        
        Args:
            detection: AprilTag detection
            
        Returns:
            Tuple of (t_vector, r_matrix) containing translation vector and rotation matrix
        """
        r_matrix = np.array(detection.pose_R).reshape(3, 3)  # Rotation matrix
        t_vector = np.array(detection.pose_t).flatten()      # Translation vector
        return t_vector, r_matrix
        
    @staticmethod
    def get_tag_id(detection):
        """
        Get AprilTag ID from detection.
        
        Args:
            detection: AprilTag detection
            
        Returns:
            Tag ID as an integer
        """
        return detection.tag_id
        
    def get_tag_world_position(self, detection):
        """
        Computes the AprilTag's position in world frame.
        
        Args:
            detection: AprilTag detection
            
        Returns:
            Tag position in world coordinates (x, y)
        """
        # Get tag pose in camera frame
        t_ca, _ = self.get_pose_from_detection(detection)
        
        # Compute the offset of the tag in camera frame
        offset = np.array([-t_ca[0] / SCALE_FACTOR, t_ca[2] / SCALE_FACTOR])
        
        return offset
    
    def get_tag_distance(self, detection):
        """
        Calculate the distance to the AprilTag.
        
        Args:
            detection: AprilTag detection
            
        Returns:
            Euclidean distance to the tag in meters
        """
        t_ca, _ = self.get_pose_from_detection(detection)
        # Calculate Euclidean distance in 3D space
        distance = np.sqrt(t_ca[0]**2 + t_ca[1]**2 + t_ca[2]**2)
        return distance