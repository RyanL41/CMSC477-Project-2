"""
Lego dropoff module for the P3-Revamp robot control system.
Handles detection, approach, and dropoff of Lego blocks at the closet.
"""
import time
import cv2
import numpy as np
from P3_Revamp.config import (
    CLOSET_LABEL, SLOW_SPEED
)
from .utilities import (
    is_centered, get_box_center, log_debug, log_info
)

class LegoDropoffController:
    def __init__(self, robot_controller, object_detector, debug=False):
        """
        Initialize the Lego dropoff controller.
        
        Args:
            robot_controller: Robot controller instance
            object_detector: Object detector instance
            debug: Enable debug mode with additional logging
        """
        self.robot = robot_controller
        self.object_detector = object_detector
        self.debug = debug
        
        # Dropoff state
        self.approach_complete = False
        self.centered = False
        
    def detect_closet(self, frame=None):
        """
        Detect the closet in the frame.
        
        Args:
            frame: Camera frame (if None, will get from robot)
            
        Returns:
            Detection data for the closet, or None if not detected
        """
        # Get the latest frame if not provided
        if frame is None:
            frame = self.robot.get_frame()
            
        if frame is None:
            return None
        
        # Run YOLO detection
        detections, _ = self.object_detector.get_detections(frame)
        
        # Get detection for the closet
        closet_detection = self.object_detector.get_best_detection(CLOSET_LABEL, detections)
        
        if closet_detection and self.debug:
            log_debug(f"Detected closet with confidence {closet_detection['confidence']:.2f}", self.debug)
        
        return closet_detection
        
    def center_robot_on_closet(self, detection):
        """
        Rotate the robot to center the detected closet in the frame.
        
        Args:
            detection: Closet detection data
            
        Returns:
            True if the closet is centered, False otherwise
        """
        if detection is None:
            return False
        
        x1, _, x2, _ = detection["box"]
        box_center_x = (x1 + x2) / 2
        camera_center_x = 320  # Assuming camera width is 640px
        
        # Calculate how far off-center the closet is
        error_x = camera_center_x - box_center_x
        
        # If the error is small, the closet is centered
        if abs(error_x) < 20:
            log_info("Closet is centered")
            self.centered = True
            return True
        
        # Calculate rotation speed based on error
        # Larger error = faster rotation, but limit maximum speed
        z_vel = np.clip(-error_x * 0.05, -16, 16)
        
        # Rotate to center the closet
        if self.debug:
            log_debug(f"Rotating to center closet, error={error_x:.2f}, z_vel={z_vel:.2f}", self.debug)
        self.robot.rotate(z_vel)
        
        self.centered = False
        return False
        
    def approach_closet(self):
        """
        Move the robot forward until the closet is no longer visible (meaning we're inside).
        
        Returns:
            True if the closet is no longer visible (we're inside), False otherwise
        """
        # Get current frame
        frame = self.robot.get_frame()
        if frame is None:
            return False
        
        # Detect the closet
        closet_detection = self.detect_closet(frame)
        
        # If the closet is not visible, we're likely inside it
        if closet_detection is None:
            # Stop the robot
            self.robot.drive_speed(x=0, y=0, z=0)
            log_info("Closet no longer visible - likely inside")
            self.approach_complete = True
            return True
        
        # Calculate the size of the closet bounding box
        x1, y1, x2, y2 = closet_detection["box"]
        box_width = x2 - x1
        box_height = y2 - y1
        
        # If the closet is taking up most of the frame, we're very close
        if box_width > 500 or box_height > 350:
            # Move forward slowly
            if self.debug:
                log_debug(f"Very close to closet (width={box_width}, height={box_height}), moving slowly", self.debug)
            self.robot.drive_speed(x=SLOW_SPEED/2, y=0, z=0)
        else:
            # Move forward at normal speed
            if self.debug:
                log_debug(f"Approaching closet (width={box_width}, height={box_height})", self.debug)
            self.robot.drive_speed(x=SLOW_SPEED, y=0, z=0)
        
        self.approach_complete = False
        return False
        
    def dropoff_block(self):
        """
        Execute the sequence to drop off a block.
        
        Returns:
            True if dropoff was successful
        """
        log_info("Executing dropoff sequence")
        
        # Stop the robot
        self.robot.drive_speed(x=0, y=0, z=0)
        time.sleep(0.5)  # Short pause to ensure the robot is fully stopped
        
        # Release the block
        log_info("Releasing block")
        self.robot.release()
        
        # Back up slightly
        log_info("Backing up")
        self.robot.backup(distance_m=0.3)
        
        log_info("Dropoff sequence complete")
        return True
        
    def lego_dropoff_loop(self):
        """
        Main lego dropoff loop.
        
        Returns:
            "success" if dropoff was successful, None otherwise
        """
        # Get current frame
        frame = self.robot.get_frame()
        if frame is None:
            return None
        
        # Detect the closet
        closet_detection = self.detect_closet(frame)
        
        # If the closet is not visible, we might be already inside or need to search for it
        if closet_detection is None:
            # Check if we've already completed the approach (inside the closet)
            if self.approach_complete:
                # Drop off the block
                dropoff_success = self.dropoff_block()
                
                # Reset state for next dropoff
                self.centered = False
                self.approach_complete = False
                
                # Return success if dropoff was successful
                if dropoff_success:
                    return "success"
            else:
                # Move forward a bit to check if we can see the closet
                log_info("Closet not visible, moving forward to search")
                self.robot.move(x=0.1)
                
                # Get new frame
                frame = self.robot.get_frame()
                closet_detection = self.detect_closet(frame)
                
                # If we still don't see the closet after moving, we might need to search
                if closet_detection is None:
                    log_info("Still can't see closet, may need to search")
                    # In a real implementation, we would add a search pattern here
            
            return None
        
        # If not centered on closet yet, do that first
        if not self.centered:
            is_centered = self.center_robot_on_closet(closet_detection)
            
            # If not centered, we're not ready to approach
            if not is_centered:
                return None
        
        # Once centered, approach the closet
        if not self.approach_complete:
            is_inside = self.approach_closet()
            
            # If not inside yet, continue approaching
            if not is_inside:
                return None
        
        # If we're centered and inside the closet, drop off the block
        dropoff_success = self.dropoff_block()
        
        # Reset state for next dropoff
        self.centered = False
        self.approach_complete = False
        
        # Return success if dropoff was successful
        if dropoff_success:
            return "success"
        
        return None
        
    def search_for_closet(self, max_rotations=24):
        """
        Rotate in place to search for the closet.
        
        Args:
            max_rotations: Maximum number of rotation steps to perform
            
        Returns:
            Closet detection if found, None otherwise
        """
        log_info("Searching for closet by rotating in place")
        
        for i in range(max_rotations):
            # Get current frame
            frame = self.robot.get_frame()
            if frame is None:
                continue
            
            # Try to detect the closet
            closet_detection = self.detect_closet(frame)
            
            # If we found the closet, stop rotating
            if closet_detection:
                log_info("Found closet during rotation search")
                return closet_detection
            
            # Rotate by a small amount
            self.robot.rotate(16)
            
            # Log progress
            if self.debug and i % 4 == 0:
                log_debug(f"Rotation search step {i+1}/{max_rotations}", self.debug)
        
        log_info("Completed full rotation without finding closet")
        return None