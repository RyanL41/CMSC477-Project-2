"""
Robot controller module for interfacing with the RoboMaster EP robot.
"""
import time
import numpy as np
from robomaster import robot, camera
from P3_Revamp.config import (
    CUBE_SIZE_METERS, INITIAL_POSITION, DEFAULT_SPEED,
    SLOW_SPEED, GRIPPER_POWER, GRIPPER_DELAY,
    ARM_UP_POSITION, ARM_DOWN_POSITION
)

class RobotController:
    def __init__(self, robot_sn, debug=False):
        """
        Initialize the robot controller.
        
        Args:
            robot_sn: Serial number of the robot
            debug: Enable debug mode with additional logging
        """
        self.robot_sn = robot_sn
        self.ep_robot = robot.Robot()
        self.debug = debug
        
        # Robot position tracking variables
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0  # Yaw angle in radians
        self.theta_offset = None
        self.x_offset = None
        self.y_offset = None
        self.pitch = 0.0
        self.roll = 0.0
        self.yaw = 0.0
        self.last_position_update = time.time()
        self.last_attitude_update = time.time()
        
        # AprilTag-based localization variables
        self.last_apriltag_update = 0
        self.last_detected_tag_id = None
        self.last_tag_relative_position = None
        self.apriltag_position_correction = (0, 0)
        
        # Grid-related variables
        self.start_grid_x = 0
        self.start_grid_y = 0
        
        # Movement control variables
        self.movement_enabled = True
        
    def initialize(self):
        """Initialize the robot and set up callbacks."""
        print(f"Initializing robot with SN: {self.robot_sn}")
        self.ep_robot.initialize(conn_type="sta", sn=self.robot_sn)

        # Start video stream
        self.ep_robot.camera.start_video_stream(
            display=False, resolution=camera.STREAM_360P
        )

        # Initialize arm and gripper
        self.ep_robot.robotic_arm.move(x=0, y=ARM_DOWN_POSITION).wait_for_completed()
        self.ep_robot.gripper.open(power=GRIPPER_POWER)
        time.sleep(1)
        self.ep_robot.gripper.pause()

        # Subscribe to position and attitude updates
        self.ep_robot.chassis.sub_position(cs=0, freq=10, callback=self.position_callback)
        self.ep_robot.chassis.sub_attitude(freq=10, callback=self.attitude_callback)
        
        # Initialize position with the known starting position
        self.set_position(INITIAL_POSITION[0], INITIAL_POSITION[1], INITIAL_POSITION[2])
        
    def position_callback(self, position_info):
        """
        Callback function to handle chassis position updates.
        
        Args:
            position_info: Position information from the robot (x, y, z)
        """
        raw_x, raw_y, _ = position_info

        if self.x_offset is None:
            self.x_offset = raw_x
        if self.y_offset is None:
            self.y_offset = raw_y

        self.x = raw_x - self.x_offset
        self.y = raw_y - self.y_offset

        self.last_position_update = time.time()
        
        if self.debug:
            print(f"Position: x={self.x:.2f}, y={self.y:.2f}, theta={np.rad2deg(self.theta):.2f}°")
    
    def attitude_callback(self, attitude_info):
        """
        Callback function to handle chassis attitude updates.
        
        Args:
            attitude_info: Attitude information from the robot (pitch, roll, yaw)
        """
        self.pitch, self.roll, self.yaw = attitude_info

        if self.pitch < 0:
            self.pitch += 360
        elif self.pitch > 360:
            self.pitch -= 360

        if self.theta_offset is None:
            self.theta_offset = self.pitch

        self.last_attitude_update = time.time()
        pitch_rad = np.deg2rad(self.pitch - self.theta_offset)
        
        # Normalize to [-π, π]
        self.theta = np.arctan2(np.sin(pitch_rad), np.cos(pitch_rad))
        
        if self.debug:
            print(f"Attitude: pitch={self.pitch:.2f}°, theta={np.rad2deg(self.theta):.2f}°")
    
    def set_grid_reference(self, grid_x, grid_y):
        """
        Set the grid reference position for coordinate transformations.
        
        Args:
            grid_x: X coordinate in grid
            grid_y: Y coordinate in grid
        """
        self.start_grid_x = grid_y
        self.start_grid_y = -grid_x
    
    def set_position(self, x, y, theta):
        """
        Set the robot's position manually.
        
        Args:
            x: X position in meters
            y: Y position in meters
            theta: Heading angle in radians
        """
        # Get the current raw position from the robot
        raw_x, raw_y, _ = self.ep_robot.chassis.get_position()
        
        # Calculate the offsets needed to make the position correct
        self.x_offset = raw_x - x
        self.y_offset = raw_y - y
        
        # Set the angle offset to make the heading correct
        current_pitch = self.pitch if self.pitch is not None else 0
        self.theta_offset = current_pitch - np.rad2deg(theta)
        
        # Update the position variables
        self.x = x
        self.y = y
        self.theta = theta
        
        print(f"Robot position manually set to: x={x:.2f}, y={y:.2f}, theta={np.rad2deg(theta):.2f}°")
    
    def update_position_from_apriltag(self, tag_id, tag_world_pos, tag_relative_pos):
        """
        Update the robot's position based on an AprilTag detection.
        
        Args:
            tag_id: ID of the detected AprilTag
            tag_world_pos: Known world position of the tag (x, y)
            tag_relative_pos: Position of the tag relative to the robot (x, y)
        """
        # Store the last tag information
        self.last_detected_tag_id = tag_id
        self.last_tag_relative_position = tag_relative_pos
        self.last_apriltag_update = time.time()
        
        # Calculate robot position based on tag position
        current_heading = self.theta
        
        # Rotation matrix to transform from robot frame to world frame
        R_z_rot = np.array([
            [np.cos(current_heading), -np.sin(current_heading)],
            [np.sin(current_heading), np.cos(current_heading)]
        ])
        
        # Calculate robot position in world frame
        robot_to_tag_vector = -np.array([tag_relative_pos[0], tag_relative_pos[1]])
        robot_to_tag_world = R_z_rot @ robot_to_tag_vector
        
        apriltag_x = tag_world_pos[0] + robot_to_tag_world[0]
        apriltag_y = tag_world_pos[1] + robot_to_tag_world[1]
        
        # Calculate position correction
        current_x, current_y, _ = self.get_position()
        correction_x = apriltag_x - current_x
        correction_y = apriltag_y - current_y
        
        # Apply the correction
        self.apriltag_position_correction = (correction_x, correction_y)
        
        # Reset odometry offsets
        raw_x, raw_y, _ = self.ep_robot.chassis.get_position()
        self.x_offset = raw_x - apriltag_x
        self.y_offset = raw_y - apriltag_y
        
        if self.debug:
            print(f"Position updated from AprilTag #{tag_id}: x={apriltag_x:.2f}, y={apriltag_y:.2f}")
        
        return apriltag_x, apriltag_y, current_heading
    
    def get_position(self):
        """
        Get the current position of the robot.
        
        Returns:
            tuple: (x, y, theta) representing position and heading in world coordinates
        """
        # Check if we have recent position data
        time_since_update = time.time() - self.last_position_update
        if time_since_update > 1.0:
            print(f"Warning: Position data is {time_since_update:.1f}s old")
        
        # Check if we have recent AprilTag detections for localization
        use_apriltag = False
        if hasattr(self, 'last_apriltag_update') and time.time() - self.last_apriltag_update < 1.0:
            use_apriltag = True
        
        # Get position with grid-based adjustments
        real_x = self.x + (self.start_grid_x * CUBE_SIZE_METERS)
        real_y = self.y + (self.start_grid_y * CUBE_SIZE_METERS)
        
        # Apply AprilTag correction if available and recent
        if use_apriltag:
            correction_x, correction_y = self.apriltag_position_correction
            real_x += correction_x
            real_y += correction_y
        
        return (real_x, real_y, self.theta)
    
    def get_frame(self):
        """
        Get the latest camera frame from the robot.
        
        Returns:
            OpenCV image or None if failed
        """
        try:
            frame = self.ep_robot.camera.read_cv2_image(strategy="newest", timeout=1.0)
            if frame is None:
                time.sleep(0.1)
            return frame
        except Exception as e:
            if self.debug:
                print(f"Error getting frame: {e}")
            time.sleep(0.5)
            return None
    
    def move(self, x=0, y=0, z=0, speed=DEFAULT_SPEED, timeout=15):
        """
        Move the robot in the specified direction.
        
        Args:
            x: Forward/backward distance in meters
            y: Left/right distance in meters
            z: Rotation angle in degrees
            speed: Movement speed
            timeout: Maximum time to wait for completion
        
        Returns:
            True if movement completed, False if not enabled or failed
        """
        if not self.movement_enabled:
            print("Movement disabled, ignoring move command")
            return False
            
        try:
            self.ep_robot.chassis.move(
                x=x, y=y, z=z, xy_speed=speed
            ).wait_for_completed(timeout=timeout)
            return True
        except Exception as e:
            print(f"Movement error: {e}")
            return False
    
    def drive_speed(self, x=0, y=0, z=0):
        """
        Drive the robot at a specified speed.
        
        Args:
            x: Forward/backward speed (-1.0 to 1.0)
            y: Left/right speed (-1.0 to 1.0)
            z: Rotation speed (-100 to 100 degrees/s)
            
        Returns:
            True if command sent, False if not enabled
        """
        if not self.movement_enabled:
            print("Movement disabled, ignoring drive command")
            return False
            
        try:
            self.ep_robot.chassis.drive_speed(x=x, y=y, z=z)
            return True
        except Exception as e:
            print(f"Drive speed error: {e}")
            return False
    
    def rotate(self, angle_deg, speed=DEFAULT_SPEED):
        """
        Rotate the robot by the specified angle in degrees.
        
        Args:
            angle_deg: Rotation angle in degrees
            speed: Movement speed
            
        Returns:
            True if rotation completed, False if not enabled or failed
        """
        if not self.movement_enabled:
            print("Movement disabled, ignoring rotate command")
            return False
            
        try:
            self.ep_robot.chassis.move(
                x=0, y=0, z=angle_deg, xy_speed=speed
            ).wait_for_completed(timeout=10)
            return True
        except Exception as e:
            print(f"Rotation error: {e}")
            return False
    
    def move_to_position(self, target_x, target_y, speed=DEFAULT_SPEED):
        """
        Move the robot to the specified position.
        
        Args:
            target_x: Target x position in meters
            target_y: Target y position in meters
            speed: Movement speed
            
        Returns:
            True if movement completed, False if not enabled or failed
        """
        if not self.movement_enabled:
            print("Movement disabled, ignoring move_to_position command")
            return False
            
        try:
            current_x, current_y, _ = self.get_position()
            dx = target_x - current_x
            dy = target_y - current_y
            
            # First, rotate to face the target
            angle = np.rad2deg(np.arctan2(dy, dx))
            current_angle = np.rad2deg(self.theta)
            angle_diff = angle - current_angle
            
            # Normalize angle difference to [-180, 180]
            while angle_diff > 180:
                angle_diff -= 360
            while angle_diff < -180:
                angle_diff += 360
                
            if abs(angle_diff) > 5:
                self.rotate(angle_diff)
            
            # Then, move forward
            distance = np.sqrt(dx**2 + dy**2)
            if distance > 0.05:  # Only move if distance is significant
                self.move(x=distance, speed=speed)
                
            return True
        except Exception as e:
            print(f"Move to position error: {e}")
            return False
    
    def grab(self):
        """
        Close the gripper to grab an object.
        
        Returns:
            True if successful
        """
        try:
            # Make sure arm is in down position
            self.ep_robot.robotic_arm.move(x=0, y=ARM_DOWN_POSITION).wait_for_completed()
            
            # Close gripper
            self.ep_robot.gripper.close(power=GRIPPER_POWER)
            time.sleep(GRIPPER_DELAY)
            self.ep_robot.gripper.pause()

            # Raise arm with object
            self.ep_robot.robotic_arm.move(x=0, y=ARM_UP_POSITION).wait_for_completed()
            
            return True
        except Exception as e:
            print(f"Grab error: {e}")
            return False
    
    def release(self):
        """
        Open the gripper to release an object.
        
        Returns:
            True if successful
        """
        try:
            # Lower arm for dropoff
            self.ep_robot.robotic_arm.move(x=0, y=ARM_DOWN_POSITION).wait_for_completed()
            
            # Open gripper
            self.ep_robot.gripper.open(power=GRIPPER_POWER)
            time.sleep(GRIPPER_DELAY)
            self.ep_robot.gripper.pause()
            
            # Raise arm
            self.ep_robot.robotic_arm.move(x=0, y=ARM_UP_POSITION).wait_for_completed()
            
            return True
        except Exception as e:
            print(f"Release error: {e}")
            return False
    
    def move_arm(self, y_distance):
        """
        Move the arm vertically by the specified distance.
        
        Args:
            y_distance: Vertical distance to move arm
            
        Returns:
            True if successful
        """
        try:
            self.ep_robot.robotic_arm.move(x=0, y=y_distance).wait_for_completed(timeout=10)
            return True
        except Exception as e:
            print(f"Move arm error: {e}")
            return False
    
    def backup(self, distance_m=0.3, speed=DEFAULT_SPEED):
        """
        Back up by the specified distance.
        
        Args:
            distance_m: Distance to back up in meters
            speed: Movement speed
            
        Returns:
            True if successful
        """
        return self.move(x=-distance_m, speed=speed)
    
    def set_movement_enabled(self, enabled):
        """
        Enable or disable robot movement.
        
        Args:
            enabled: Boolean indicating whether movement is enabled
        """
        self.movement_enabled = enabled
        if not enabled:
            # Stop the robot when disabling movement
            self.ep_robot.chassis.drive_speed(x=0, y=0, z=0)
            print("Robot movement disabled")
        else:
            print("Robot movement enabled")
    
    def cleanup(self):
        """
        Clean up robot resources.
        
        Should be called when shutting down.
        """
        try:
            # Stop any movement
            self.ep_robot.chassis.drive_speed(x=0, y=0, z=0)
            
            # Unsubscribe from callbacks
            self.ep_robot.chassis.unsub_position()
            self.ep_robot.chassis.unsub_attitude()
            
            # Stop video stream
            self.ep_robot.camera.stop_video_stream()
            
            # Close connection
            self.ep_robot.close()
            
            print("Robot resources cleaned up")
        except Exception as e:
            print(f"Cleanup error: {e}")