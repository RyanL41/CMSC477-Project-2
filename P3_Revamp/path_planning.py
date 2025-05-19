"""
Path planning module for the P3-Revamp robot control system.
Implements target-oriented path planning with obstacle avoidance.
"""
import time
import cv2
import numpy as np
from P3_Revamp.config import (
    ROTATION_SPEED, DEFAULT_SPEED, SLOW_SPEED, MAX_SINGLE_MOVE_DISTANCE,
    DISTANCE_THRESHOLD, ROTATION_SEARCH_INTERVAL, SEARCH_ROTATION_STEP,
    OBSTACLE_AVOIDANCE_DISTANCE, APRILTAG_PROXIMITY_THRESHOLD
)
from P3_Revamp.utilities import (
    calculate_distance, calculate_angle_to_target, 
    is_target_reached, is_centered, get_rotation_direction,
    log_debug, log_info
)

class PathPlanner:
    def __init__(self, robot_controller, object_detector, apriltag_detector, debug=False):
        """
        Initialize the path planner.
        
        Args:
            robot_controller: Robot controller instance
            object_detector: Object detector instance
            apriltag_detector: AprilTag detector instance
            debug: Enable debug mode with additional logging
        """
        self.robot = robot_controller
        self.object_detector = object_detector
        self.apriltag_detector = apriltag_detector
        self.debug = debug
        
        # Path planning variables
        self.target_labels = []
        self.target_coordinate = None
        self.last_rotation_search = 0
        self.path = []
        self.target_found = False
        
    def set_target(self, target_labels, target_coordinate, distance_threshold=DISTANCE_THRESHOLD):
        """
        Set the target labels and coordinate for path planning.
        
        Args:
            target_labels: List of YOLO labels to look for
            target_coordinate: (x, y) coordinate to move towards
            distance_threshold: How close to get to target coordinate
            
        Returns:
            Dictionary with target information
        """
        if isinstance(target_labels, str):
            target_labels = [target_labels]
            
        self.target_labels = target_labels
        self.target_coordinate = target_coordinate
        self.target_found = False
        
        target_info = {
            'labels': target_labels,
            'coordinate': target_coordinate,
            'distance_threshold': distance_threshold,
            'last_seen': None
        }
        
        log_info(f"Target set to labels: {target_labels} at coordinate: {target_coordinate}")
        
        return target_info
        
    def rotate_to_face_target(self, target_coordinate=None):
        """
        Rotate the robot to face the target coordinate.
        
        Args:
            target_coordinate: (x, y) coordinate to face (if None, uses self.target_coordinate)
            
        Returns:
            True if the rotation is complete
        """
        if target_coordinate is None:
            target_coordinate = self.target_coordinate
            
        if target_coordinate is None:
            log_debug("No target coordinate provided", self.debug)
            return False
        
        # Get current position and heading
        print("Robot.get_postition:",self.robot.get_position())
        current_x, current_y, current_heading = self.robot.get_position()
        
        # Calculate angle to target
        target_x, target_y = target_coordinate
        dx = target_x - current_x
        dy = target_y - current_y
        
        # Calculate angle in radians
        target_angle = np.arctan2(dy, dx)
        
        # Convert to degrees
        target_angle_deg = np.rad2deg(target_angle)
        current_heading_deg = np.rad2deg(current_heading)
        
        # Calculate angle difference
        print("ANGLE DIFF:",(target_angle_deg - current_heading_deg) % 360)
        angle_diff = (target_angle_deg - current_heading_deg) % 360
        if angle_diff > 180:
            angle_diff -= 360
        
        log_debug(f"Angle to target: {target_angle_deg:.2f}°, current heading: {current_heading_deg:.2f}°, difference: {angle_diff:.2f}°", self.debug)
        
        # If the difference is small enough, we're already facing the target
        if abs(angle_diff) < 5:
            return True
        
        # Rotate to face the target
        self.robot.rotate(angle_diff)
        
        return True
        
    def move_toward_target(self, target_coordinate=None, speed=DEFAULT_SPEED):
        """
        Move the robot toward the target coordinate.
        
        Args:
            target_coordinate: (x, y) coordinate to move toward (if None, uses self.target_coordinate)
            speed: Movement speed
            
        Returns:
            True if movement was executed, False otherwise
        """
        if target_coordinate is None:
            target_coordinate = self.target_coordinate
            
        if target_coordinate is None:
            log_debug("No target coordinate provided", self.debug)
            return False
        
        # Get current position
        current_x, current_y, _ = self.robot.get_position()
        current_pos = (current_x, current_y)
        
        # Get target coordinate
        target_x, target_y = target_coordinate
        
        # Calculate distance to target
        distance = calculate_distance(current_pos, target_coordinate)
        
        # If we're close enough to the target, we're done
        if distance < DISTANCE_THRESHOLD:
            log_info(f"Reached target coordinate: {target_coordinate}")
            return False
        
        # Calculate movement vector (move in small increments)
        move_distance = min(distance, MAX_SINGLE_MOVE_DISTANCE)
        
        # First, rotate to face the target
        self.rotate_to_face_target(target_coordinate)
        
        # Move forward
        log_debug(f"Moving toward target: distance={move_distance:.2f}m", self.debug)
        self.robot.move(x=move_distance, speed=speed)
        
        return True
        
    def check_for_target_objects(self, target_labels=None, frame=None):
        """
        Check if any target objects are visible in the current frame.
        
        Args:
            target_labels: List of target labels to look for (if None, uses self.target_labels)
            frame: Camera frame (if None, will get from robot)
            
        Returns:
            (found_object, detection) if target object is found, (False, None) otherwise
        """
        if target_labels is None:
            target_labels = self.target_labels
            
        if not target_labels:
            log_debug("No target labels provided", self.debug)
            return False, None
        
        # Get the latest frame if not provided
        if frame is None:
            frame = self.robot.get_frame()
            
        if frame is None:
            return False, None
        
        # Run object detection
        detections, _ = self.object_detector.get_detections(frame)
        
        if not detections:
            return False, None
        
        # Check if we see any of the target labels
        for target_label in target_labels:
            detection = self.object_detector.get_best_detection(target_label, detections)
            if detection:
                log_info(f"Found target object: {target_label}")
                self.target_found = True
                return True, detection
        
        return False, None
        
    def search_for_target_objects(self, target_labels=None, max_rotations=24):
        """
        Rotate in place to search for target objects when they're not visible.
        
        Args:
            target_labels: List of target labels to look for (if None, uses self.target_labels)
            max_rotations: Maximum number of rotation steps to perform
            
        Returns:
            (found_object, detection) if target object is found, (False, None) otherwise
        """
        if target_labels is None:
            target_labels = self.target_labels
            
        log_info(f"Searching for objects: {target_labels} by rotating in place")
        
        for i in range(max_rotations):
            # Try to find target objects
            found_object, detection = self.check_for_target_objects(target_labels)
            
            # If we found a target object, stop rotating
            if found_object:
                log_info(f"Found target object during rotation search")
                return True, detection
            
            # Rotate by a small amount
            self.robot.rotate(SEARCH_ROTATION_STEP)
            
            # Log progress
            if self.debug and i % 4 == 0:
                log_debug(f"Rotation search step {i+1}/{max_rotations}", self.debug)
        
        log_info("Completed full rotation without finding target objects")
        return False, None
        
    def check_for_obstacles(self, frame=None):
        """
        Check for obstacles in the current frame.
        
        Args:
            frame: Camera frame (if None, will get from robot)
            
        Returns:
            (has_obstacle, obstacle_type, obstacle_detection) if obstacle detected,
            (False, None, None) otherwise
        """
        # Get the latest frame if not provided
        if frame is None:
            frame = self.robot.get_frame()
            
        if frame is None:
            return False, None, None
        
        # Run YOLO detection to find other robots
        detections, _ = self.object_detector.get_detections(frame)
        robot_detection = self.object_detector.get_best_detection("robot", detections)
        
        if robot_detection:
            log_debug("Robot obstacle detected", self.debug)
            return True, "robot", robot_detection
        
        # Convert frame to grayscale for AprilTag detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect AprilTags
        apriltag_detections = self.apriltag_detector.find_tags(gray)
        
        if apriltag_detections:
            # For simplicity, just handle the first AprilTag
            detection = apriltag_detections[0]
            
            # Get tag position relative to camera
            tag_relative_pos = self.apriltag_detector.get_tag_world_position(detection)
            tag_distance = np.linalg.norm(tag_relative_pos)
            
            # If the tag is close enough (<threshold box units)
            if tag_distance < APRILTAG_PROXIMITY_THRESHOLD:
                tag_id = self.apriltag_detector.get_tag_id(detection)
                log_debug(f"AprilTag #{tag_id} obstacle detected at distance {tag_distance:.2f}", self.debug)
                return True, "apriltag", detection
        
        return False, None, None
        
    def avoid_obstacle(self, obstacle_type, obstacle_detection):
        """
        Issue movement commands for obstacle avoidance.
        
        Args:
            obstacle_type: Type of obstacle ("robot" or "apriltag")
            obstacle_detection: Detection data for the obstacle
            
        Returns:
            True if avoidance was successful, False otherwise
        """
        if obstacle_type == "robot":
            x1, _, x2, _ = obstacle_detection["box"]
            box_center_x = (x1 + x2) / 2
            camera_center_x = 320  # Assuming camera width is 640px
            
            if box_center_x < camera_center_x:
                # Robot is on the left, move right
                log_info("Avoiding robot obstacle by moving right")
                self.robot.move(x=0, y=-OBSTACLE_AVOIDANCE_DISTANCE, z=0)
            else:
                # Robot is on the right, move left
                log_info("Avoiding robot obstacle by moving left")
                self.robot.move(x=0, y=OBSTACLE_AVOIDANCE_DISTANCE, z=0)
            
            return True
            
        elif obstacle_type == "apriltag":
            # Get tag position relative to camera
            tag_relative_pos = self.apriltag_detector.get_tag_world_position(obstacle_detection)
            
            if tag_relative_pos[0] < 0:
                # Tag is on the left, move right
                log_info("Avoiding AprilTag obstacle by moving right")
                self.robot.move(x=0, y=-OBSTACLE_AVOIDANCE_DISTANCE, z=0)
            else:
                # Tag is on the right, move left
                log_info("Avoiding AprilTag obstacle by moving left")
                self.robot.move(x=0, y=OBSTACLE_AVOIDANCE_DISTANCE, z=0)
            
            return True
        
        return False
        
    def path_planning_loop(self):
        """
        Main path planning loop.
        
        Returns:
            (found_object, detection) if target object is found,
            (False, None) otherwise
        """
        # First check for obstacles
        has_obstacle, obstacle_type, obstacle_detection = self.check_for_obstacles()
        
        if has_obstacle:
            # Handle obstacle avoidance
            self.avoid_obstacle(obstacle_type, obstacle_detection)
            return False, None
        
        # Check if target is already reached
        current_pos = self.robot.get_position()
        if is_target_reached(current_pos, self.target_coordinate):
            log_info(f"Reached target coordinate: {self.target_coordinate}")
            
            # Perform rotation search at the target position
            found_object, detection = self.search_for_target_objects()
            return found_object, detection
        
        # Check for target objects in current view
        found_object, detection = self.check_for_target_objects()
        if found_object:
            return True, detection
        
        # If we don't see any target objects and haven't searched recently,
        # perform a periodic rotation search
        current_time = time.time()
        if current_time - self.last_rotation_search > ROTATION_SEARCH_INTERVAL:
            log_debug("Performing periodic rotation search", self.debug)
            found_object, detection = self.search_for_target_objects()
            self.last_rotation_search = current_time
            if found_object:
                return True, detection
        
        # If target not found and not at target position, move toward target
        self.move_toward_target()
        
        return False, None