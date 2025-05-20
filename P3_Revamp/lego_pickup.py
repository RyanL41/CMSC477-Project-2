"""
Enhanced Lego pickup module with memory of previously seen blocks.

This module provides an enhanced Lego block detection and pickup system that maintains
memory of previously seen blocks across frames. It implements:

1. Block memory system to track blocks over time
2. Target selection based on consistency and confidence
3. Target locking to maintain focus on a specific block
4. Scanning behavior to find blocks that aren't currently visible
5. Obstacle avoidance during pickup

The system prioritizes blocks that have been consistently detected across multiple
frames, improving robustness against detection noise and temporary occlusions.
"""
import time
import cv2
import numpy as np
from P3_Revamp.config import (
    BLOCK_LABELS, LEGO_PICKUP_Y_THRESHOLD, SLOW_SPEED,
    ARM_DOWN_POSITION, ARM_UP_POSITION, GRIPPER_POWER, GRIPPER_DELAY
)
from P3_Revamp.utilities import (
    is_centered, get_box_center, log_debug, log_info
)

class LegoPickupController:
    """Base Lego pickup controller class."""
    
    def __init__(self, robot_controller, object_detector, apriltag_detector, debug=False):
        """
        Initialize the basic Lego pickup controller.
        
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
        If no detection is available, rotate to search for blocks.
        
        Args:
            detection: Block detection data
            label: Block label (for logging)
            
        Returns:
            True if the block is centered, False otherwise
        """
        print("Step 1")
        if detection is None:
            # If no detection, rotate to search for blocks
            if self.debug:
                log_debug("No detection available, rotating to search", self.debug)
            # Rotate at a moderate speed
            self.robot.drive_speed(x=0, y=0, z=-16)  # Positive z value means counterclockwise rotation
            return False
        
        print("Step 2")
        x1, _, x2, _ = detection["box"]
        box_center_x = (x1 + x2) / 2
        camera_center_x = 320  # Assuming camera width is 640px
        
        print("Step 3")
            # Calculate how far off-center the block is
        error_x = camera_center_x - box_center_x
        
        print("Step 4")
        # If the error is small, the block is centered
        if abs(error_x) < 20:
            if label:
                log_info(f"Block {label} is centered")
            self.centered = True
            return True
        
        print("Step 5")
        # Calculate rotation speed based on error
        # Larger error = faster rotation, but limit maximum speed
        z_vel = np.clip(-error_x * 0.05, -16, 16)
        
        print("Step 6")
        # Rotate to center the block
        if self.debug:
            log_debug(f"Rotating to center block, error={error_x:.2f}, z_vel={z_vel:.2f}", self.debug)
        self.robot.drive_speed(x=0, y=0, z=z_vel)
        
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
                self.robot.move(x=0, y=0.3, z=0)
            else:
                # Robot is on the right, move left
                log_info("Avoiding robot obstacle by moving left")
                self.robot.move(x=0, y=-0.3, z=0)
            
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
                
                return False
        
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


class EnhancedLegoPickupController(LegoPickupController):
    """Enhanced Lego pickup controller with memory capabilities."""
    
    def __init__(self, robot_controller, object_detector, apriltag_detector, debug=False):
        """
        Initialize the enhanced Lego pickup controller with memory of previously seen blocks.
        
        Args:
            robot_controller: Robot controller instance
            object_detector: Object detector instance
            apriltag_detector: AprilTag detector instance
            debug: Enable debug mode with additional logging
        """
        super().__init__(robot_controller, object_detector, apriltag_detector, debug)
        
        # Replace current_target_label with target_label for consistency
        del self.current_target_label
        
        # Memory of seen blocks
        self.seen_blocks = {}  # label -> {timestamp, position, confidence, consistency}
        self.target_label = None
        self.target_lock = False
        self.memory_timeout = 10.0  # Seconds to remember blocks
        self.consistency_threshold = 3  # Detections needed to consider a block consistent
        
        # Last known position tracking
        self.last_known_detection = None
        self.last_known_label = None
        self.last_known_timestamp = 0
        
    def detect_lego_blocks(self, frame=None):
        """
        Detect Lego blocks in the frame and update memory.
        
        Args:
            frame: Camera frame (if None, will get from robot)
            
        Returns:
            Dictionary with detected blocks by label
        """
        # Call the parent method to get base detections
        detected_blocks = super().detect_lego_blocks(frame)
        
        # Add memory-specific processing
        current_time = time.time()
        
        # Filter out old blocks from memory
        self._clean_memory(current_time)
        
        # Update memory with current detections
        for label, detection in detected_blocks.items():
            self._update_memory(label, detection, current_time)
        
        if self.debug and self.seen_blocks:
            memory_info = ", ".join([f"{l}({self.seen_blocks[l]['consistency']})" for l in self.seen_blocks])
            log_debug(f"Block memory: {memory_info}", self.debug)
        
        return detected_blocks
    
    def _clean_memory(self, current_time):
        """
        Remove blocks from memory that haven't been seen recently.
        
        Args:
            current_time: Current timestamp
        """
        labels_to_remove = []
        
        for label, data in self.seen_blocks.items():
            if current_time - data['timestamp'] > self.memory_timeout:
                labels_to_remove.append(label)
        
        for label in labels_to_remove:
            if label != self.target_label:  # Don't remove target
                del self.seen_blocks[label]
                if self.debug:
                    log_debug(f"Forgot block {label} due to timeout", self.debug)
    
    def _update_memory(self, label, detection, current_time):
        """
        Update memory with a new detection.
        
        Args:
            label: Block label
            detection: Detection data
            current_time: Current timestamp
        """
        # Get detection center
        center_x, center_y = get_box_center(detection)
        robot_pos = self.robot.get_robot_position()
        
        # Calculate position relative to robot
        # This is a simplification - a real implementation would do proper coordinate transforms
        rel_x = center_x - 320  # Distance from center of image
        rel_y = detection['box'][3]  # Bottom of bounding box (y2)
        
        if label in self.seen_blocks:
            # Update existing entry
            old_data = self.seen_blocks[label]
            
            # Update position with exponential moving average
            alpha = 0.3  # Weight for new position
            old_data['position']['rel_x'] = (1-alpha) * old_data['position']['rel_x'] + alpha * rel_x
            old_data['position']['rel_y'] = (1-alpha) * old_data['position']['rel_y'] + alpha * rel_y
            
            # Update confidence and timestamp
            old_data['confidence'] = max(old_data['confidence'], detection['confidence'])
            old_data['timestamp'] = current_time
            
            # Increase consistency counter
            old_data['consistency'] += 1
            
            # Cap consistency at a maximum value
            if old_data['consistency'] > 10:
                old_data['consistency'] = 10
        else:
            # Create new entry
            self.seen_blocks[label] = {
                'position': {'rel_x': rel_x, 'rel_y': rel_y},
                'robot_pos': robot_pos,
                'confidence': detection['confidence'],
                'timestamp': current_time,
                'consistency': 1
            }
    
    def select_target_block(self):
        """
        Select the most consistent block as target.
        
        Returns:
            Selected target label or None
        """
        # If we already have a locked target, keep using it
        if self.target_lock and self.target_label in self.seen_blocks:
            return self.target_label
        
        # Find block with highest consistency
        best_label = None
        best_consistency = 0
        best_confidence = 0
        
        for label, data in self.seen_blocks.items():
            # Only consider blocks that have been seen consistently
            if data['consistency'] < self.consistency_threshold:
                continue
                
            # Prefer more consistent blocks, but use confidence as tiebreaker
            if (data['consistency'] > best_consistency or 
                (data['consistency'] == best_consistency and data['confidence'] > best_confidence)):
                best_label = label
                best_consistency = data['consistency']
                best_confidence = data['confidence']
        
        # Lock onto this target if found
        if best_label is not None:
            self.target_label = best_label
            self.target_lock = True
            log_info(f"Locked onto target block: {best_label} (consistency: {best_consistency}, confidence: {best_confidence:.2f})")
        
        return best_label
    
    def find_best_block_in_frame(self, detected_blocks):
        """
        Find the best block in the current frame, with preference to the closest block to the previously detected one.
        Also updates the last known detection information.
        
        Args:
            detected_blocks: Dictionary of detected blocks by label
            
        Returns:
            (detection, label) of the best block, or (None, None) if no blocks detected
        """
        if not detected_blocks:
            # When no blocks detected, don't reset last_known_detection/label here
            # This is handled separately to allow movement toward last position
            return None, None
            
        # If we have a previous detection, find the closest block to that one
        if self.last_known_detection is not None:
            # Get position of last known detection
            last_x1, last_y1, last_x2, last_y2 = self.last_known_detection["box"]
            last_center_x = (last_x1 + last_x2) / 2
            last_center_y = (last_y1 + last_y2) / 2
            
            # Find the block closest to the last known position
            closest_label = None
            closest_detection = None
            min_distance = float('inf')
            
            for label, detection in detected_blocks.items():
                x1, y1, x2, y2 = detection["box"]
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                
                # Calculate distance between centers
                distance = np.sqrt((center_x - last_center_x)**2 + (center_y - last_center_y)**2)
                
                if distance < min_distance:
                    min_distance = distance
                    closest_label = label
                    closest_detection = detection
            
            if closest_label is not None:
                log_debug(f"Locked onto block {closest_label} (closest to last detected block)", self.debug)
                
                # Update last known detection
                self.last_known_detection = closest_detection
                self.last_known_label = closest_label
                self.last_known_timestamp = time.time()
                self.target_lock = True  # Lock onto this block
                self.target_label = closest_label
                
                return closest_detection, closest_label
        
        # If we don't have a previous detection or couldn't find a match,
        # fall back to closest to center
        detection, label = self.find_closest_block_to_center(detected_blocks)
        
        if detection and label:
            log_debug(f"Using closest block to center: {label}", self.debug)
            
            # Save as last known detection
            self.last_known_detection = detection
            self.last_known_label = label 
            self.last_known_timestamp = time.time()
            self.target_lock = True  # Lock onto this block
            self.target_label = label
        
        return detection, label
    
    def scan_for_target(self, max_rotations=24):
        """
        Scan in a circle looking for the target block.
        
        Args:
            max_rotations: Maximum rotation steps
            
        Returns:
            (detection, label) if target found, (None, None) otherwise
        """
        # If no target selected, can't scan for it
        if not self.target_label:
            return None, None
            
        log_info(f"Scanning for target block: {self.target_label}")
        
        # Calculate step size based on max rotations
        step_size = 360 / max_rotations
        
        for i in range(max_rotations):
            # Get frame and detect blocks
            frame = self.robot.get_frame()
            if frame is None:
                continue
                
            # Run detection
            detections, _ = self.object_detector.get_detections(frame)
            detection = self.object_detector.get_best_detection(self.target_label, detections)
            
            if detection:
                log_info(f"Found target block {self.target_label} during scan")
                return detection, self.target_label
            
            # Rotate by step size
            self.robot.rotate(step_size)
            
            # Log progress
            if self.debug and i % 4 == 0:
                log_debug(f"Scan rotation {i+1}/{max_rotations}", self.debug)
        
        log_info(f"Could not find target block {self.target_label} during scan")
        
        # If we couldn't find the target after a full scan, unlock it
        self.target_lock = False
        self.target_label = None
        
        return None, None
    
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
        
        log_info(f"Pickup sequence complete for block {self.target_label}")
        return True
    
    def lego_pickup_loop(self):
        """
        Main lego pickup loop with memory of previously seen blocks.
        
        Returns:
            "success:label" if pickup was successful, None otherwise
        """
        # First, check for obstacles
        obstacle_avoided = self.check_for_obstacles_during_pickup()

        print("Step 1")
        
        # If we had to avoid an obstacle, skip the rest of the loop
        if obstacle_avoided:
            return None
        
        # Get current frame
        frame = self.robot.get_frame()
        if frame is None:
            return None

        print("Step 2")
        
        # Detect lego blocks and update memory
        detected_blocks = self.detect_lego_blocks(frame)

        print("Step 3")
        
        # If we don't have a target yet, try to select one
        if not self.target_label or not self.target_lock:
            self.select_target_block()
        
        # Find the best block in the current frame
        best_detection, best_label = self.find_best_block_in_frame(detected_blocks)

        print("Step 4")

        # If no blocks were detected but we have a target, try to scan for it
        if best_detection is None and self.target_label:
            best_detection, best_label = self.scan_for_target()

        print("Step 5")
        
        # As requested, immediately reset the lock and last detection when no blocks detected
        if best_detection is None:
            # First, attempt one final drive command using the last known position
            if self.last_known_detection is not None and time.time() - self.last_known_timestamp < 1.0:
                log_info(f"No blocks detected. Making final movement toward last position of {self.last_known_label} before reset")
                
                # Get the position from the last known detection
                x1, y1, x2, y2 = self.last_known_detection["box"]
                box_center_x = (x1 + x2) / 2
                camera_center_x = 320  # Assuming camera width is 640px
                
                # Estimate y (left-right) from horizontal position in image
                error_x = camera_center_x - box_center_x
                y_vel = np.clip(error_x * 0.005, -0.1, 0.1)
                z_vel = np.clip(-error_x * 0.05, -16, 16)
                
                # Move forward with small adjustments
                self.robot.drive_speed(x=0.1, y=y_vel, z=z_vel)
            else:
                # Just rotate to search
                self.robot.drive_speed(x=0, y=0, z=-16)
                
            # Reset all tracking variables as requested
            log_info("Resetting block tracking - no blocks detected in current frame")
            self.last_known_detection = None
            self.last_known_label = None
            self.last_known_timestamp = 0
            self.target_lock = False
            self.target_label = None
            
            return None
        
        # # If we haven't centered on the block yet, do that first
        if not self.centered:
            print(self.centered)
            is_centered = self.center_robot_on_block(best_detection, best_label)
            
            # If not centered, we're not ready to approach
            if not is_centered:
                return None
            
            # Robot is now centered on the block
            # Lower the arm to prepare for pickup
            log_info("Lowering arm to prepare for pickup")
            self.robot.ep_robot.robotic_arm.move(x=0, y=ARM_DOWN_POSITION).wait_for_completed()
            # Make sure gripper is open
            self.robot.ep_robot.gripper.open(power=GRIPPER_POWER)
            time.sleep(0.5)
            self.robot.ep_robot.gripper.pause()
        
        # Once centered, approach the block if not already at pickup position
        if not self.approach_complete:
            # Get the latest frame and detection
            frame = self.robot.get_frame()
            if frame is None:
                return None

            print("Step 6")
            
            detected_blocks = self.detect_lego_blocks(frame)
            
            # Try to get the target block, or the best available block
            if self.target_label and self.target_label in detected_blocks:
                best_detection = detected_blocks[self.target_label]
                best_label = self.target_label
            else:
                best_detection, best_label = self.find_best_block_in_frame(detected_blocks)
            
            if best_detection is None:
                # Lost track of the block, need to reset and find it again
                self.centered = False
                return None
            
            print("Step 7")
            
            # Update target if needed
            if best_label != self.target_label and best_label is not None:
                self.target_label = best_label
                self.target_lock = True
                log_info(f"Updated target to {best_label}")
            
            print("Step 8")
            
            # Approach the block
            is_at_pickup_position = self.approach_block(best_detection, best_label)
            
            print("Step 9")
            
            # If not at pickup position yet, continue approaching
            if not is_at_pickup_position:
                return None
        
        # If we're centered and at pickup position, pick up the block
        pickup_success = self.pickup_block()
        
        print("Step 10")

        # Save target_label for return value
        target_label = self.target_label
        
        # Reset state for next pickup
        self.centered = False
        self.approach_complete = False
        self.target_lock = False
        self.target_label = None
        self.seen_blocks = {}  # Clear memory after pickup
        
        # Return success if pickup was successful
        if pickup_success:
            return f"success:{target_label}"
        
        return None