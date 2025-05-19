"""
Lego pickup module for the P3-Revamp robot control system.
Handles detection, approach, and pickup of Lego blocks.
"""
import time
import cv2
import numpy as np
from P3_Revamp.config import (
    BLOCK_LABELS, LEGO_PICKUP_Y_THRESHOLD, SLOW_SPEED,
    ARM_DOWN_POSITION, ARM_UP_POSITION
)
from P3_Revamp.utilities import (
    is_centered, get_box_center, log_debug, log_info
)

class LegoPickupController:
    def __init__(self, robot_controller, object_detector, apriltag_detector, debug=False):
        """
        Initialize the Lego pickup controller.
        
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
        
        # Pickup state
        self.current_target_label = None
        self.approach_complete = False
        self.centered = False
        
    def detect_lego_blocks(self, frame=None):
        """
        Detect Lego blocks in the frame.
        
        Args:
            frame: Camera frame (if None, will get from robot)
            
        Returns:
            Dictionary with detected blocks by label
        """
        # Get the latest frame if not provided
        if frame is None:
            frame = self.robot.get_frame()
            
        if frame is None:
            return {}
        
        # Run YOLO detection
        detections, _ = self.object_detector.get_detections(frame)
        
        # Get detections for each lego block type
        detected_blocks = {}
        for label in BLOCK_LABELS:
            detection = self.object_detector.get_best_detection(label, detections)
            if detection:
                detected_blocks[label] = detection
        
        if self.debug:
            log_debug(f"Detected {len(detected_blocks)} lego blocks: {list(detected_blocks.keys())}", self.debug)
        
        return detected_blocks
        
    def find_closest_block_to_center(self, detected_blocks):
        """
        Find the block closest to the center of the camera frame.
        
        Args:
            detected_blocks: Dictionary of detected blocks by label
            
        Returns:
            (detection, label) of the closest block, or (None, None) if no blocks detected
        """
        if not detected_blocks:
            return None, None
        
        best_detection = None
        best_label = None
        min_center_distance = float('inf')
        
        for label, detection in detected_blocks.items():
            # Get center distance
            center_distance = detection.get('center_distance', float('inf'))
            
            if center_distance < min_center_distance:
                min_center_distance = center_distance
                best_detection = detection
                best_label = label
        
        if self.debug and best_label:
            log_debug(f"Closest block to center: {best_label} with distance {min_center_distance:.2f}", self.debug)
        
        return best_detection, best_label
        
    def center_robot_on_block(self, detection, label=None):
        """
        Rotate the robot to center the detected block in the frame.
        
        Args:
            detection: Block detection data
            label: Block label (for logging)
            
        Returns:
            True if the block is centered, False otherwise
        """
        if detection is None:
            return False
        
        x1, _, x2, _ = detection["box"]
        box_center_x = (x1 + x2) / 2
        camera_center_x = 320  # Assuming camera width is 640px
        
        # Calculate how far off-center the block is
        error_x = camera_center_x - box_center_x
        
        # If the error is small, the block is centered
        if abs(error_x) < 20:
            if label:
                log_info(f"Block {label} is centered")
            self.centered = True
            return True
        
        # Calculate rotation speed based on error
        # Larger error = faster rotation, but limit maximum speed
        z_vel = np.clip(-error_x * 0.05, -15, 15)
        
        # Rotate to center the block
        if self.debug:
            log_debug(f"Rotating to center block, error={error_x:.2f}, z_vel={z_vel:.2f}", self.debug)
        self.robot.rotate(z_vel)
        
        self.centered = False
        return False
        
    def approach_block(self, detection, label=None):
        """
        Move the robot forward until the block is at the optimal pickup position.
        
        Args:
            detection: Block detection data
            label: Block label (for logging)
            
        Returns:
            True if the block is at the optimal position for pickup, False otherwise
        """
        if detection is None:
            return False
        
        # Get the y-coordinate of the bottom of the bounding box
        _, y1, _, y2 = detection["box"]
        
        # If the block is already at the optimal position
        if y1 > LEGO_PICKUP_Y_THRESHOLD:
            if label:
                log_info(f"Block {label} is at optimal pickup position (y1={y1})")
            
            # Stop the robot
            self.robot.drive_speed(x=0, y=0, z=0)
            self.approach_complete = True
            return True
        
        # Calculate forward speed based on distance from target position
        # Slow down as we get closer to the target
        distance_to_target = LEGO_PICKUP_Y_THRESHOLD - y1
        
        # Use slower speed when close to target
        if distance_to_target < 50:
            x_vel = SLOW_SPEED / 2
        else:
            x_vel = SLOW_SPEED
            
        if self.debug:
            log_debug(f"Approaching block, y1={y1}, distance_to_target={distance_to_target:.2f}, x_vel={x_vel:.2f}", self.debug)
        
        # Move forward
        self.robot.drive_speed(x=x_vel, y=0, z=0)
        
        self.approach_complete = False
        return False
        
    def pickup_block(self):
        """
        Execute the sequence to pick up a block.
        
        Returns:
            True if pickup was successful
        """
        log_info("Executing pickup sequence")
        
        # Stop the robot
        self.robot.drive_speed(x=0, y=0, z=0)
        time.sleep(0.5)  # Short pause to ensure the robot is fully stopped
        
        # Lower the arm
        log_info("Lowering arm")
        self.robot.move_arm(ARM_DOWN_POSITION)
        
        # Close the gripper
        log_info("Closing gripper")
        self.robot.grab()
        
        log_info("Pickup sequence complete")
        return True
        
    def check_for_obstacles_during_pickup(self, frame=None):
        """
        Check for obstacles during the pickup process and avoid them if necessary.
        
        Args:
            frame: Camera frame (if None, will get from robot)
            
        Returns:
            True if obstacles were detected and avoided, False otherwise
        """
        # Get the latest frame if not provided
        if frame is None:
            frame = self.robot.get_frame()
            
        if frame is None:
            return False
        
        # Run YOLO detection to find other robots
        detections, _ = self.object_detector.get_detections(frame)
        robot_detection = self.object_detector.get_best_detection("robot", detections)
        
        if robot_detection:
            x1, _, x2, _ = robot_detection["box"]
            box_center_x = (x1 + x2) / 2
            camera_center_x = 320  # Assuming camera width is 640px
            
            if box_center_x < camera_center_x:
                # Robot is on the left, move right
                log_info("Avoiding robot obstacle by moving right")
                self.robot.move(x=0, y=-0.3, z=0)
            else:
                # Robot is on the right, move left
                log_info("Avoiding robot obstacle by moving left")
                self.robot.move(x=0, y=0.3, z=0)
            
            return True
        
        # Convert frame to grayscale for AprilTag detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect AprilTags
        apriltag_detections = self.apriltag_detector.find_tags(gray)
        
        if apriltag_detections:
            # For simplicity, just handle the first AprilTag
            detection = apriltag_detections[0]
            
            # Get tag position relative to camera
            tag_relative_pos = self.apriltag_detector.get_tag_world_position(detection)
            
            # If the tag is close enough (< 1 box unit or ~0.26m)
            if np.linalg.norm(tag_relative_pos) < 1.0:
                if tag_relative_pos[0] < 0:
                    # Tag is on the left, move right
                    log_info("Avoiding AprilTag obstacle by moving right")
                    self.robot.move(x=0, y=-0.3, z=0)
                else:
                    # Tag is on the right, move left
                    log_info("Avoiding AprilTag obstacle by moving left")
                    self.robot.move(x=0, y=0.3, z=0)
                
                return True
        
        return False
        
    def lego_pickup_loop(self):
        """
        Main lego pickup loop.
        
        Returns:
            "success" if pickup was successful, None otherwise
        """
        # First, check for obstacles
        obstacle_avoided = self.check_for_obstacles_during_pickup()
        
        # If we had to avoid an obstacle, skip the rest of the loop
        if obstacle_avoided:
            return None
        
        # Get current frame
        frame = self.robot.get_frame()
        if frame is None:
            return None
        
        # Detect lego blocks
        detected_blocks = self.detect_lego_blocks(frame)
        
        # Find the closest block to the center
        best_detection, best_label = self.find_closest_block_to_center(detected_blocks)
        
        # If no blocks were detected, return None
        if best_detection is None:
            return None
        
        # Save current target label
        self.current_target_label = best_label
        
        # If we haven't centered on the block yet, do that first
        if not self.centered:
            is_centered = self.center_robot_on_block(best_detection, best_label)
            
            # If not centered, we're not ready to approach
            if not is_centered:
                return None
        
        # Once centered, approach the block if not already at pickup position
        if not self.approach_complete:
            # Get the latest frame and detection
            frame = self.robot.get_frame()
            if frame is None:
                return None
            
            detected_blocks = self.detect_lego_blocks(frame)
            best_detection = detected_blocks.get(self.current_target_label)
            
            if best_detection is None:
                # Lost track of the block, need to reset and find it again
                self.centered = False
                return None
            
            # Approach the block
            is_at_pickup_position = self.approach_block(best_detection, self.current_target_label)
            
            # If not at pickup position yet, continue approaching
            if not is_at_pickup_position:
                return None
        
        # If we're centered and at pickup position, pick up the block
        pickup_success = self.pickup_block()
        
        # Reset state for next pickup
        self.centered = False
        self.approach_complete = False
        self.current_target_label = None
        
        # Return success if pickup was successful
        if pickup_success:
            return "success"
        
        return None