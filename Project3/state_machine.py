"""
State machine implementation for robot control.
"""
import threading
import time
import cv2
import numpy as np
import traceback


from Project3.config import (
    SCALE_FACTOR, RobotState, STARTING_POSITION_NUMBER, SELF_CLOSET_NUMBER, 
    TARGET_CLOSET_NUMBER, CAMERA_MATRIX, APRILTAG_SIZE_METERS
)
from Project3.apriltag_detector import AprilTagDetector
from Project3.vision import ObjectDetector
from Project3.robot_controller import RobotController
from Project3.grid import load_grid_from_csv, process_grid, find_position_in_grid, grid_to_world_coords

# Import path planning 
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from djikstra import get_path


LEGO_BIG_LABEL = "lego_big"
LEGO_SMALL_LABEL = "lego_small"
LEGO_MED_LABEL = "lego_med"

class RobotStateMachine:
    def __init__(self, robot_sn, map_file):
        """
        Initialize the state machine.
        
        Args:
            robot_sn: Serial number of the robot
            map_file: Path to the CSV map file
        """
        self.map_file = map_file
        self.current_state = RobotState.INITIALIZING
        
        # Initialize robot controller
        self.robot = RobotController(robot_sn)
        
        # Initialize vision systems
        self.object_detector = ObjectDetector()
        self.apriltag_detector = AprilTagDetector(
            np.array(CAMERA_MATRIX), 
            marker_size_m=APRILTAG_SIZE_METERS
        )
        
        # Target tracking
        self.target_label = None
        self.last_detection = None
        
        # Load and process grid
        self.grid = load_grid_from_csv(map_file)
        self.processed_grid = process_grid(self.grid)
        
        # Map configuration
        self.starting_pos_number = STARTING_POSITION_NUMBER
        self.self_closet_number = SELF_CLOSET_NUMBER
        self.target_closet_number = TARGET_CLOSET_NUMBER
        
        # Performance tracking data
        self.approach_data = {
            state.value: {
                "time_steps": [],
                "actual_x": [],
                "target_x": [],
                "actual_y": [],
                "target_y": [],
            }
            for state in RobotState
        }

    def initialize(self):
        """Initialize the robot and setup systems."""
        self.robot.initialize()
        
        # Find robot starting position in grid
        start_x, start_y = find_position_in_grid(self.grid, self.starting_pos_number)
        if start_x is not None and start_y is not None:
            self.robot.set_grid_reference(start_x, start_y)
            print(f"Robot starting position: grid ({start_x}, {start_y})")
        else:
            print("Warning: No starting position found in grid, using (0,0)")
        
        self.current_state = RobotState.LOOKING_FOR_BLOCK_IN_CLOSET

    def get_closet_position(self, closet_number):
        """Find the coordinates of a closet in the grid."""
        grid_x, grid_y = find_position_in_grid(self.grid, closet_number)
        return grid_to_world_coords(grid_x, grid_y)
    
    def get_path_from_vision(self):


        while True:
            current_x, current_y, current_heading = self.robot.get_position()
            current_x_blocks, current_y_blocks = current_x / SCALE_FACTOR, current_y / SCALE_FACTOR

            # Get camera frame and run detections
            frame = self.robot.get_frame()
            if frame is None:
                continue
            
            # Run YOLO detection
            detections, _ = self.object_detector.get_detections(frame)
            self.target_label = LEGO_SMALL_LABEL
            found_small_object = self.object_detector.get_best_detection(self.target_label, detections)
            self.target_label = LEGO_MED_LABEL
            found_med_object = self.object_detector.get_best_detection(self.target_label, detections)
            self.target_label = LEGO_BIG_LABEL
            found_big_object = self.object_detector.get_best_detection(self.target_label, detections)


            # Run AprilTag detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            apriltag_detections = self.apriltag_detector.find_tags(gray)
            
            # Process AprilTag detections
            if apriltag_detections:
                print(f"Found {len(apriltag_detections)} AprilTags")
                robot_pos = (current_x, current_y)
                
                for detection in apriltag_detections:
                    # Get tag world position
                    tag_relative_pos = self.apriltag_detector.get_tag_world_position(detection)

                    # rotate the relative position by the robot's heading
                    R_z_rot = np.array(
                        [[np.cos(current_heading), -np.sin(current_heading)], [np.sin(current_heading), np.cos(current_heading)]]
                    )
                    print("tag_relative_pos",tag_relative_pos)
                    tag_relative_pos = R_z_rot @ tag_relative_pos
                    print("tag_relative_pos",tag_relative_pos)
                    tag_world_pos = (current_x_blocks + tag_relative_pos[0], current_y_blocks + tag_relative_pos[1])
                    
                    # Get tag ID
                    tag_id = self.apriltag_detector.get_tag_id(detection)
                    print(f"AprilTag ID {tag_id} at position: ({tag_world_pos[0]:.3f}, {tag_world_pos[1]:.3f})")
            
    

    def follow_path(self):
        """Find blocks in the closet and approach them."""
    
        if self.path is not None:
            
            for waypoint in self.path:

                current_pos = self.robot.get_position()
                
                R_z_rot = np.array(
                    [[np.cos(current_pos[2]), -np.sin(current_pos[2])], [np.sin(current_pos[2]), np.cos(current_pos[2])]]
                )

                current_pos_blocks = [
                    current_pos[0] / 0.26,
                    current_pos[1] / 0.26
                ]

                target_pos_blocks = [
                    waypoint[0],
                    -waypoint[1]
                ]
                move_x = target_pos_blocks[0] - current_pos_blocks[0]
                move_y = target_pos_blocks[1] - current_pos_blocks[1]
                
                vel = np.array([move_x, move_y])

                vel = R_z_rot @ vel
                
                vel *= 0.26

                print(f"Moving to waypoint: dx={vel[0]:.3f}, dy={vel[1]:.3f} theta={current_pos[2]:.3f}")
                self.robot.ep_robot.chassis.move(x=vel[0], y=vel[1],z=0, xy_speed=0.1).wait_for_completed()

    def handle_approach_block(self,object):
        self.robot.approach(self,object)

    def handle_grab_block(self):
        """Grab a block with the gripper."""
        self.robot.grab()

    def handle_drop_off(self):
        """Release a block with the gripper."""
        self.robot.release()

    def handle_move_arm(self, y_distance):
        """Move the robot arm."""
        self.robot.move_arm(y_distance)

    def handle_backup(self, distance_m=0.3):
        """Back up the robot."""
        self.robot.backup(distance_m)

    def run(self):
        """Main state machine loop."""
        # Initialize if needed
        if self.current_state == RobotState.INITIALIZING:
            self.initialize()
        
        # Main control loop
        while self.current_state != RobotState.ERROR:
            if cv2.waitKey(1) == ord("q"):
                self.robot.cleanup()
                break

            # Print current state
            current_time = time.strftime("%Y-%m-%d %H:%M:%S")
            print(f"\n--- {current_time} | State: {self.current_state.value} ---")

            try:
                # Handle states
                if self.current_state == RobotState.LOOKING_FOR_BLOCK_IN_CLOSET:
                    # In parallel, we call follow_path_to_closet and update_path_with_vision
                    # 2 async loops
                    # follow_path_to_closet looks at self.path and follows it
                    # one async loop with get_path_from_vision()
                    # one async loop with follow_path()
                    self.path = get_path(self.grid, self.starting_pos_number, self.self_closet_number, upscaling_factor=2, num_points=50)

                    # Create and start two threads
                    vision_thread = threading.Thread(target=self.get_path_from_vision)
                    path_thread = threading.Thread(target=self.follow_path)
                    
                    vision_thread.start()
                    path_thread.start()
                    
                    # Wait for the follow_path thread to complete
                    path_thread.join()
                    
                    # Stop the vision thread
                    self.stop_vision_thread = True
                    vision_thread.join()

                elif self.current_state == RobotState.APPROACH_BLOCK:
                    self.handle_approach_block()
                elif self.current_state == RobotState.GRAB_BLOCK:
                    self.handle_grab_block()
                elif self.current_state == RobotState.DROP_OFF:
                    self.handle_drop_off()
                elif self.current_state == RobotState.MOVE_ARM:
                    self.handle_move_arm(-100)  # Default move up
                elif self.current_state == RobotState.BACKUP:
                    self.handle_backup()
                # Add other states as needed
                
            except Exception as e:
                print(traceback.format_exc())
                self.current_state = RobotState.ERROR
                self.robot.cleanup()
                break

        print(f"\n=== State Machine Finished with State: {self.current_state.value} ===")
        self.robot.cleanup()
