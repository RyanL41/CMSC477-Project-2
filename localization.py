"""
Localization module for the P3-Revamp robot control system.
Handles AprilTag detection and tracking for robot localization.
"""
import time
import cv2
import numpy as np
from P3_Revamp.config import (
    ROTATION_SPEED, INITIAL_POSITION, APRILTAG_PROXIMITY_THRESHOLD
)
from P3_Revamp.utilities import (
    preprocess_frame, log_debug, log_info, 
    calculate_distance, safe_set_apriltag
)

class Localizer:
    def __init__(self, robot_controller, apriltag_detector, debug=False):
        """
        Initialize the localizer.
        
        Args:
            robot_controller: Robot controller instance
            apriltag_detector: AprilTag detector instance
            debug: Enable debug mode with additional logging
        """
        self.robot = robot_controller
        self.detector = apriltag_detector
        self.debug = debug
        self.apriltag_positions = {}  # Tag ID -> {position, last_seen}
        self.initialized = False
        
    def initialize_localization(self):
        """
        Initialize localization with known starting position.
        
        Returns:
            Boolean indicating whether initialization was successful
        """
        # Set initial known position
        x, y, theta = INITIAL_POSITION
        self.robot.set_position(x, y, theta)
        log_info(f"Initialized robot position to: x={x:.2f}, y={y:.2f}, theta={np.rad2deg(theta):.2f}°")
        
        self.initialized = True
        return True
        
    def detect_and_track_apriltags(self, frame=None, shared_data=None, lock=None):
        """
        Detect and track AprilTags in the provided frame.
        
        Args:
            frame: Camera frame (if None, will get from robot)
            shared_data: Optional shared data structure to update
            lock: Optional thread lock for shared data
            
        Returns:
            List of AprilTag detections or None if no detections
        """
        # Get the latest frame if not provided
        if frame is None:
            frame = self.robot.get_frame()
            
        if frame is None:
            return None
        
        # Preprocess frame
        frame = preprocess_frame(frame)
        if frame is None:
            return None
        
        # Convert to grayscale for AprilTag detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect AprilTags
        detections = self.detector.find_tags(gray)
        
        if not detections:
            log_debug("No AprilTags detected", self.debug)
            return None
        
        # Get current robot position
        current_x, current_y, current_heading = self.robot.get_robot_position()
        
        # Process each detected AprilTag
        for detection in detections:
            # Get tag ID
            tag_id = self.detector.get_tag_id(detection)
            
            # Get tag position relative to robot
            tag_relative_pos = self.detector.get_tag_world_position(detection)
            
            # Convert to world coordinates
            R_z_rot = np.array([
                [np.cos(current_heading), -np.sin(current_heading)],
                [np.sin(current_heading), np.cos(current_heading)]
            ])
            
            tag_relative_pos = R_z_rot @ tag_relative_pos
            tag_world_pos = (current_x + tag_relative_pos[1], current_y - tag_relative_pos[0])
            
            # Store in hash table
            tag_data = {
                'position': tag_world_pos,
                'relative_position': tag_relative_pos,
                'last_seen': time.time()
            }
            
            # Update local hash table
            self.apriltag_positions[tag_id] = tag_data
            
            # Update shared data if provided
            if shared_data is not None and lock is not None:
                safe_set_apriltag(shared_data, tag_id, tag_data, lock)
            
            log_debug(f"AprilTag ID {tag_id} at position: ({tag_world_pos[0]:.3f}, {tag_world_pos[1]:.3f})", self.debug)
            
            # Update robot position based on AprilTag if we've seen this tag before
            if hasattr(self.robot, 'update_position_from_apriltag'):
                self.robot.update_position_from_apriltag(tag_id, tag_world_pos, tag_relative_pos)
        
        return detections
    
    def search_for_apriltags(self, max_rotations=24):
        """
        Rotate in place to find AprilTags if none are visible.
        
        Args:
            max_rotations: Maximum number of rotation steps to perform
            
        Returns:
            True if AprilTags were found, False otherwise
        """
        log_info("Searching for AprilTags by rotating in place")
        
        for i in range(max_rotations):
            # Try to detect AprilTags
            detections = self.detect_and_track_apriltags()
            
            # If we found an AprilTag, stop rotating
            if detections:
                log_info(f"Found {len(detections)} AprilTags during rotation search")
                return True
            
            # Rotate by a small amount
            self.robot.rotate(ROTATION_SPEED)
            
            # Log progress
            if self.debug and i % 4 == 0:
                log_debug(f"Rotation search step {i+1}/{max_rotations}", self.debug)
        
        log_info("Completed full rotation without finding AprilTags")
        return False
    
    def localization_loop(self):
        """
        Main localization loop.
        
        Returns:
            True if localization is successful, False otherwise
        """
        if not self.initialized:
            self.initialize_localization()
        
        # Try to detect AprilTags
        detections = self.detect_and_track_apriltags()
        
        # If no AprilTags were detected, search for them
        if not detections:
            found = self.search_for_apriltags()
            if not found:
                log_info("Warning: No AprilTags found during 360-degree scan")
                return False
        
        # Successfully localized
        return True
    
    def is_apriltag_too_close(self, tag_id=None):
        """
        Check if an AprilTag is too close to the robot.
        
        Args:
            tag_id: Optional specific tag ID to check
            
        Returns:
            True if an AprilTag is too close, False otherwise
        """
        # Check if we have any AprilTag positions
        if not self.apriltag_positions:
            return False
        
        current_time = time.time()
        
        # If tag_id provided, check only that tag
        if tag_id is not None:
            if tag_id in self.apriltag_positions:
                tag_data = self.apriltag_positions[tag_id]
                # Check if tag is recent
                if current_time - tag_data['last_seen'] > 1.0:
                    return False
                
                # Check distance to tag
                tag_rel_pos = tag_data['relative_position']
                distance = np.linalg.norm(tag_rel_pos)
                return distance < APRILTAG_PROXIMITY_THRESHOLD
            return False
        
        # Check all tags
        for tag_id, tag_data in self.apriltag_positions.items():
            # Check if tag is recent
            if current_time - tag_data['last_seen'] > 1.0:
                continue
            
            # Check distance to tag
            tag_rel_pos = tag_data['relative_position']
            distance = np.linalg.norm(tag_rel_pos)
            if distance < APRILTAG_PROXIMITY_THRESHOLD:
                return True
        
        return False
    
    def get_closest_apriltag(self):
        """
        Get the closest AprilTag to the robot.
        
        Returns:
            Tuple of (tag_id, distance) or (None, None) if no tags
        """
        if not self.apriltag_positions:
            return None, None
        
        current_time = time.time()
        closest_tag_id = None
        min_distance = float('inf')
        
        for tag_id, tag_data in self.apriltag_positions.items():
            # Check if tag is recent
            if current_time - tag_data['last_seen'] > 1.0:
                continue
            
            # Check distance to tag
            tag_rel_pos = tag_data['relative_position']
            distance = np.linalg.norm(tag_rel_pos)
            
            if distance < min_distance:
                min_distance = distance
                closest_tag_id = tag_id
        
        if closest_tag_id is None:
            return None, None
        
        return closest_tag_id, min_distance