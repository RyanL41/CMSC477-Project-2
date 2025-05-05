"""
Robot controller module for interfacing with the RoboMaster EP robot.
"""
import time
import numpy as np
from robomaster import robot, camera
from Project3.config import CUBE_SIZE_METERS

class RobotController:
    def __init__(self, robot_sn):
        """
        Initialize the robot controller.
        
        Args:
            robot_sn: Serial number of the robot
        """
        self.robot_sn = robot_sn
        self.ep_robot = robot.Robot()
        
        # Robot position tracking variables
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0  # Yaw angle in degrees
        self.theta_offset = None
        self.pitch = 0.0
        self.roll = 0.0
        self.yaw = 0.0
        self.last_position_update = time.time()
        self.last_attitude_update = time.time()
        
        # Grid-related variables
        self.start_grid_x = 0
        self.start_grid_y = 0
        
    def initialize(self):
        """Initialize the robot and set up callbacks."""
        print(f"Initializing robot with SN: {self.robot_sn}")
        self.ep_robot.initialize(conn_type="sta", sn=self.robot_sn)

        # Start video stream
        self.ep_robot.camera.start_video_stream(
            display=False, resolution=camera.STREAM_360P
        )

        # Initialize arm and gripper
        self.ep_robot.robotic_arm.move(x=0, y=-100).wait_for_completed()
        self.ep_robot.gripper.open(power=70)
        time.sleep(1)
        self.ep_robot.gripper.pause()

        # Subscribe to position and attitude updates
        self.ep_robot.chassis.sub_position(cs=0, freq=5, callback=self.position_callback)
        self.ep_robot.chassis.sub_attitude(freq=5, callback=self.attitude_callback)
        
    def position_callback(self, position_info):
        """Callback function to handle chassis position updates."""
        self.x, self.y, _ = position_info
        self.last_position_update = time.time()
        print(f"Position: x={self.x:.2f}, y={self.y:.2f}, theta={self.theta:.2f}°")
    
    def attitude_callback(self, attitude_info):
        """Callback function to handle chassis attitude updates."""
        self.pitch, self.roll, self.yaw = attitude_info

        if self.pitch < 0:
            self.pitch += 360
        elif self.pitch > 360:
            self.pitch -= 360

        if self.theta_offset is None:
            self.theta_offset = self.pitch
            print("Offset: ", self.theta_offset)

        self.last_attitude_update = time.time()
        self.theta = self.pitch - self.theta_offset

        if self.theta > 360:
            self.theta -= 360
        elif self.theta < 0:
            self.theta += 360
        
        self.theta = np.deg2rad(self.theta)
        #print(f"Attitude: pitch={self.pitch:.2f}°, roll={self.roll:.2f}°, yaw={self.yaw:.2f}°")
    
    def set_grid_reference(self, grid_x, grid_y):
        """Set the grid reference position for coordinate transformations."""
        self.start_grid_x = grid_y
        self.start_grid_y = -grid_x
    
    def get_position(self):
        """Returns the latest position of the robot with grid-based adjustments."""
        time_since_update = time.time() - self.last_position_update
        if time_since_update > 1.0:
            print(f"Warning: Stale position data ({time_since_update:.1f}s old)")
        
        # Adjust coordinates based on grid starting position
        real_x = self.x + (self.start_grid_x * CUBE_SIZE_METERS)
        real_y = self.y + (self.start_grid_y * CUBE_SIZE_METERS)
        
        return (real_x, real_y, self.theta)
    
    def get_frame(self):
        """Get the latest camera frame from the robot."""
        try:
            frame = self.ep_robot.camera.read_cv2_image(strategy="newest", timeout=1.0)
            if frame is None:
                time.sleep(0.1)
            return frame
        except Exception:
            time.sleep(0.5)
            return None
    
    def move(self, x=0, y=0, z=0, speed=0.3):
        """Move the robot in the specified direction."""
        self.ep_robot.chassis.move(
            x=x, y=y, z=z, xy_speed=speed
        ).wait_for_completed(timeout=15)
    
    def rotate(self, angle_deg):
        """Rotate the robot by the specified angle in degrees."""
        self.ep_robot.chassis.move(
            x=0, y=0, z=angle_deg
        ).wait_for_completed(timeout=10)
    
    def move_to_position(self, target_x, target_y):
        """Move the robot to the specified position."""
        current_x, current_y, _ = self.get_position()
        dx = target_x - current_x
        dy = target_y - current_y
        
        # First, rotate to face the target
        angle = np.rad2deg(np.arctan2(dy, dx))
        angle_diff = angle - self.theta
        self.rotate(angle_diff)
        
        # Then, move forward
        distance = np.sqrt(dx**2 + dy**2)
        self.move(x=distance)
    
    def grab(self):
        """Close the gripper to grab an object."""
        self.ep_robot.gripper.close(power=70)
        time.sleep(1.5)
        self.ep_robot.gripper.pause()
    
    def release(self):
        """Open the gripper to release an object."""
        self.ep_robot.gripper.open(power=70)
        time.sleep(1.5)
        self.ep_robot.gripper.pause()
    
    def move_arm(self, y_distance):
        """Move the arm vertically by the specified distance."""
        self.ep_robot.robotic_arm.move(y=y_distance).wait_for_completed(timeout=10)
    
    def backup(self, distance_m=0.3):
        """Back up by the specified distance."""
        self.move(x=-distance_m)
    
    def cleanup(self):
        """Clean up robot resources."""
        self.ep_robot.chassis.unsub_position()
        self.ep_robot.chassis.unsub_attitude()
        self.ep_robot.camera.stop_video_stream()
        self.ep_robot.close()
