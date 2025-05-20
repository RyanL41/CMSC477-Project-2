"""
Center line handling module for the P3-Revamp robot control system.
Handles detection, centering, and following of the center line.
"""
import time
import cv2
import numpy as np
from P3_Revamp.config import (
    CENTER_LINE_LABEL, SLOW_SPEED, CENTER_THRESHOLD
)
from P3_Revamp.utilities import (
    is_centered, get_box_center, log_debug, log_info
)

class CenterLineController:
    def __init__(self, robot_controller, object_detector, debug=False):
        """
        Initialize the center line controller.
        
        Args:
            robot_controller: Robot controller instance
            object_detector: Object detector instance
            debug: Enable debug mode with additional logging
        """
        self.robot = robot_controller
        self.object_detector = object_detector
        self.debug = debug
        
        # Center line state
        self.centered = False
        self.waypoint_reached = False
        self.saved_position = None
        
    def detect_center_line(self, frame=None):
        """
        Detect the center line in the frame.
        
        Args:
            frame: Camera frame (if None, will get from robot)
            
        Returns:
            Detection data for the center line, or None if not detected
        """
        # Get the latest frame if not provided
        if frame is None:
            frame = self.robot.get_frame()
            
        if frame is None:
            return None
        
        # Run YOLO detection
        detections, _ = self.object_detector.get_detections(frame)
        
        # Get detection for the center line
        center_line_detection = self.object_detector.get_best_detection(CENTER_LINE_LABEL, detections)
        
        if center_line_detection and self.debug:
            log_debug(f"Detected center line with confidence {center_line_detection['confidence']:.2f}", self.debug)
        
        return center_line_detection
        
    def rotate_to_face_center_line(self, detection):
        """
        Rotate the robot to face the detected center line.
        
        Args:
            detection: Center line detection data
            
        Returns:
            True if the robot is facing the center line, False otherwise
        """
        if detection is None:
            return False
        
        x1, y1, x2, y2 = detection["box"]
        box_center_x = (x1 + x2) / 2
        camera_center_x = 320  # Assuming camera width is 640px
        
        # Calculate how far off-center the line is
        error_x = camera_center_x - box_center_x
        
        # If the error is small, the robot is facing the center line
        if abs(error_x) < CENTER_THRESHOLD:
            log_info("Robot is facing the center line")
            self.centered = True
            return True
        
        # Calculate rotation speed based on error
        z_vel = np.clip(-error_x * 0.05, -15, 15)
        
        # Rotate to face the center line
        if self.debug:
            log_debug(f"Rotating to face center line, error={error_x:.2f}, z_vel={z_vel:.2f}", self.debug)
        self.robot.rotate(z_vel)
        
        self.centered = False
        return False
        
    def center_on_line(self):
        """
        Center the robot on the center line by translating laterally.
        
        Returns:
            True if the robot is centered on the line, False otherwise
        """
        # First, rotate to 270 degrees (facing down)
        current_x, current_y, current_heading = self.robot.get_robot_position()
        target_heading = 270  # degrees
        
        # Convert current heading to degrees
        current_heading_deg = np.rad2deg(current_heading)
        
        # Calculate angle difference
        angle_diff = (target_heading - current_heading_deg) % 360
        if angle_diff > 180:
            angle_diff -= 360
        
        # Rotate to face down
        if abs(angle_diff) > 5:
            log_info(f"Rotating to 270 degrees, current={current_heading_deg:.2f}, diff={angle_diff:.2f}")
            self.robot.rotate(angle_diff)
            return False
        
        # Get current frame
        frame = self.robot.get_frame()
        if frame is None:
            return False
        
        # Detect the center line
        center_line_detection = self.detect_center_line(frame)
        
        # If the center line is not visible, we can't center on it
        if center_line_detection is None:
            log_info("Center line not visible, can't center")
            return False
        
        # Calculate the center of the center line
        x1, _, x2, _ = center_line_detection["box"]
        box_center_x = (x1 + x2) / 2
        camera_center_x = 320  # Assuming camera width is 640px
        
        # Calculate how far off-center the line is
        error_x = camera_center_x - box_center_x
        
        # If the error is small, the robot is centered on the line
        if abs(error_x) < 10:
            # Stop the robot
            self.robot.drive_speed(x=0, y=0, z=0)
            log_info("Robot is centered on the center line")
            
            # Save the current position
            self.save_target_coordinate()
            return True
        
        # Calculate lateral movement speed based on error
        y_vel = np.clip(error_x * 0.001, -0.1, 0.1)
        
        # Translate laterally to center on the line
        if self.debug:
            log_debug(f"Centering on line, error={error_x:.2f}, y_vel={y_vel:.2f}", self.debug)
        self.robot.drive_speed(x=0, y=y_vel, z=0)
        
        return False
        
    def save_target_coordinate(self):
        """
        Save the current position as a target coordinate for future navigation.
        
        Returns:
            (x, y) coordinate position
        """
        current_x, current_y, _ = self.robot.get_robot_position()
        self.saved_position = (current_x, current_y)
        log_info(f"Saved target coordinate: {self.saved_position}")
        return self.saved_position
        
    def center_line_pre_waypoint_loop(self):
        """
        Main center line handling loop for pre-waypoint.
        
        Returns:
            Tuple of ("success", target_position) if successful, (None, None) otherwise
        """
        # Get current frame
        frame = self.robot.get_frame()
        if frame is None:
            return None, None
        
        # Detect the center line
        center_line_detection = self.detect_center_line(frame)
        
        # If the center line is not visible, we need to search for it
        if center_line_detection is None:
            # Rotate in place to search for the center line
            log_info("Center line not visible, rotating to search")
            self.robot.rotate(15)
            return None, None
        
        # Try to face the center line
        is_facing = self.rotate_to_face_center_line(center_line_detection)
        
        # If the robot is facing the center line, perform the centering step
        if is_facing:
            is_centered = self.center_on_line()
            
            # If the robot is centered on the line, return success and saved position
            if is_centered and self.saved_position is not None:
                return "success", self.saved_position
        
        return None, None
        
    def center_line_post_waypoint_loop(self, target_position):
        """
        Main center line handling loop for post-waypoint.
        
        Args:
            target_position: Target position to return to
            
        Returns:
            "success" if successful, None otherwise
        """
        # Calculate distance to target position
        current_x, current_y, _ = self.robot.get_robot_position()
        target_x, target_y = target_position
        
        distance = np.sqrt((target_x - current_x)**2 + (target_y - current_y)**2)
        
        # If we're close enough to the target position, perform the centering step
        if distance < 0.5:
            is_centered = self.center_on_line()
            
            # If the robot is centered on the line, we're done
            if is_centered:
                return "success"
        else:
            # Move toward the target position
            log_info(f"Moving toward saved position, distance={distance:.2f}m")
            self.robot.move_to_position(target_x, target_y)
        
        return None
        
    def search_for_center_line(self, max_rotations=24):
        """
        Rotate in place to search for the center line.
        
        Args:
            max_rotations: Maximum number of rotation steps to perform
            
        Returns:
            Center line detection if found, None otherwise
        """
        log_info("Searching for center line by rotating in place")
        
        for i in range(max_rotations):
            # Get current frame
            frame = self.robot.get_frame()
            if frame is None:
                continue
            
            # Try to detect the center line
            center_line_detection = self.detect_center_line(frame)
            
            # If we found the center line, stop rotating
            if center_line_detection:
                log_info("Found center line during rotation search")
                return center_line_detection
            
            # Rotate by a small amount
            self.robot.rotate(15)
            
            # Log progress
            if self.debug and i % 4 == 0:
                log_debug(f"Rotation search step {i+1}/{max_rotations}", self.debug)
        
        log_info("Completed full rotation without finding center line")
        return None