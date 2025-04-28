import os
import time
import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
from robomaster import robot, camera
from enum import Enum
import traceback
from matplotlib import pyplot as plt
import pupil_apriltags
import traceback
from queue import Empty
from scipy.spatial.transform import Rotation as R
from matplotlib import pyplot as plt

from djikstra import get_path


YOLO_MODEL_PATH = "best.pt"
#NEED TO DETERMINE THESE VALUES
TARGET_BBOX_SMALL_HEIGHT_APPROACH = 160
TARGET_BBOX_MEDIUM_HEIGHT_APPROACH = 125
TARGET_BBOX_LARGE_HEIGHT_APPROACH = 192


class Project3States(Enum):
    INITIALIZING = "initializing"
    LOOKING_FOR_BLOCK_IN_CLOSET = "looking_for_block_in_closet" 
    REMOVE_FROM_CLOSET = "remove_from_closet"
    LOOKING_FOR_BLOCK = "looking_for_block"
    APPROACH_BLOCK = "approach_block"
    GRAB_BLOCK = "grab_block"
    LIFT_ARM = "lift_arm"
    LOWER_ARM = "lower_arm"
    DROP_OFF = "drop_off"
    BACKUP = "backup"
    DELIVER_BLOCK = "deliver_block"
    BULLY_MODE = "bully_mode"
    WALL_MODE = "wall_mode"
    ERROR = "error"

LEGO_BIG_LABEL = "lego_big"
LEGO_SMALL_LABEL = "lego_small"
LEGO_MED_LABEL = "lego_med"

STARTING_POSITION_NUMBER = 2 # either 2 or 5 (see InitialMap.csv)
SELF_CLOSET_NUMBER = 4 # either 3 or 4 (see InitialMap.csv)
TARGET_CLOSET_NUMBER = 3 # either 3 or 4 (see InitialMap.csv)   

BLOCK_LABELS = [LEGO_BIG_LABEL, LEGO_SMALL_LABEL, LEGO_MED_LABEL]

class AprilTagDetector:
    def __init__(self, K, family="tag36h11", threads=2, marker_size_m=0.16):
        self.camera_params = [K[0, 0], K[1, 1], K[0, 2], K[1, 2]]
        self.marker_size_m = marker_size_m
        self.detector = pupil_apriltags.Detector(family, threads)
    def find_tags(self, frame_gray):
        detections = self.detector.detect(
            frame_gray,
            estimate_tag_pose=True,
            camera_params=self.camera_params,
            tag_size=self.marker_size_m,
        )
        return detections



class Project3StateMachine:
    def __init__(self, robot_sn, map_file):
        self.csv_path = map_file
        self.robot_sn = robot_sn
        self.ep_robot = robot.Robot()
        self.yolo_model = YOLO(YOLO_MODEL_PATH)

        self.current_state = Project3States.INITIALIZING
        self.last_detection = None
        self.last_vis_frame = None
        #self.positions = get_path(self.csv_path,STARTING_POSITION_NUMBER,SELF_CLOSET_NUMBER)

        K = np.array(
        [[314, 0, 320], [0, 314, 180], [0, 0, 1]]
        )  # Camera focal length and center pixel
        marker_size_m = 0.153  # Size of the AprilTag in meters
        self.apriltag = AprilTagDetector(K, threads=2, marker_size_m=marker_size_m)
            
        self.target_label = None    
        # Robot position tracking variables
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0  # Yaw angle in degrees
        self.pitch = 0.0
        self.roll = 0.0
        self.last_position_update = time.time()
        self.last_attitude_update = time.time()
            
        # Grid-related variables (to be initialized in initialize_robot)
        self.cube_size_meters = 0.26  # 1 cube unit = 0.26 meters
        self.grid = None
        self.processed_grid = None
        self.starting_pos_number = STARTING_POSITION_NUMBER
        self.self_closet_number = SELF_CLOSET_NUMBER
        self.target_closet_number = TARGET_CLOSET_NUMBER
        self.start_grid_x = 0
        self.start_grid_y = 0
        self.start_rot = 0
        
        self.approach_plot_data = {
            state.value: {
                "time_steps": [],
                "actual_x": [],
                "target_x": [],
                "actual_y": [],
                "target_y": [],
            }
            for state in Project3States
        }

    def position_callback(self, position_info):
        """Callback function to handle chassis position updates"""
        # Extract position data from the callback
        self.x, self.y, self.theta = position_info
        self.last_position_update = time.time()
        print(f"Position update: x={self.x:.2f}, y={self.y:.2f}, theta={self.theta:.2f}°")
    
    def attitude_callback(self, attitude_info):
        """Callback function to handle chassis attitude updates"""
        # Extract attitude data from the callback
        self.pitch, self.roll, self.yaw = attitude_info
        self.last_attitude_update = time.time()
        print(f"Attitude update: pitch={self.pitch:.2f}°, roll={self.roll:.2f}°, yaw={self.yaw:.2f}°")
        
    def get_position(self):
        """Returns the latest position of the robot with grid-based adjustments"""
        # Use the latest position values updated by the callbacks
        time_since_update = time.time() - self.last_position_update
        if time_since_update > 1.0:  # If position hasn't been updated in more than 1 second
            print(f"Warning: Using stale position data ({time_since_update:.1f}s old)")
        
        # Adjust coordinates based on grid starting position
        # Convert grid coordinates to real-world coordinates (1 grid unit = 0.26 meters)
        real_x = self.x + (self.start_grid_x * self.cube_size_meters)
        real_y = self.y + (self.start_grid_y * self.cube_size_meters)
        
        # Return adjusted position
        return (real_x, real_y, self.theta)
        
    def get_grid_data(self, csv_path=None):
        """
        Reads a CSV file and converts it to a numpy array.
        
        Args:
            csv_path: Path to the CSV file. If None, uses self.csv_path.
            
        Returns:
            A numpy array representation of the grid.
        """
        if csv_path is None:
            csv_path = self.csv_path
            
        df = pd.read_csv(csv_path, header=None)
        return np.array(df)
    def get_frame(self):
        try:
            frame = self.ep_robot.camera.read_cv2_image(strategy="newest", timeout=1.0)
            if frame is None:
                time.sleep(0.1)
            return frame
        except Exception as e:
            time.sleep(0.5)
            return None


    def run_yolo_detection(self, frame):
        if frame is None or self.yolo_model is None:
            return [], None

        results = self.yolo_model.predict(
            source=frame, show=False, verbose=False, conf=0.70
        )[0]

        boxes = results.boxes
        class_names = self.yolo_model.names
        vis_frame = frame.copy()
        detections_list = []

        for box in boxes:
            xyxy = box.xyxy.cpu().numpy().flatten().astype(int)
            class_id = int(box.cls.cpu().numpy())
            label = class_names[class_id]
            confidence = float(box.conf.cpu().numpy())

            detections_list.append(
                {"label": label, "confidence": confidence, "box": xyxy}
            )

            cv2.rectangle(
                vis_frame,
                (xyxy[0], xyxy[1]),
                (xyxy[2], xyxy[3]),
                color=(0, 255, 0),
                thickness=2,
            )
            label_text = f"{label} ({confidence:.2f})"
            cv2.putText(
                vis_frame,
                label_text,
                (xyxy[0], xyxy[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )

        self.last_vis_frame = vis_frame
        return detections_list, vis_frame


    def get_target_label_detection(self, target_label, detections):

        if not detections:
            return None

        best_detection = None
        max_confidence = 0.0

        for det in detections:
            if det["label"] == target_label and det["confidence"] > max_confidence:
                max_confidence = det["confidence"]
                best_detection = det

        return best_detection

    def get_padded_grid(self, grid, radius):
        """
        Adds padding around obstacles in the grid.
        
        Args:
            grid: The grid to pad.
            radius: The radius around obstacles to pad.
            
        Returns:
            A padded grid.
        """
        padded_grid = np.copy(grid)
        obstacles = np.where(grid == 1)
        for i, j in zip(obstacles[0], obstacles[1]):
            # calculate all cells in the radius
            for x in range(max(0, i - radius), min(len(grid), i + radius + 1)):
                for y in range(max(0, j - radius), min(len(grid[i]), j + radius + 1)):
                    padded_grid[x][y] = 1
        
        return padded_grid
    


    def upscale_grid(self, grid=None, upscaling_factor=4):
        """
        Upscales a grid by the given factor.
        
        Args:
            grid: The grid to upscale. If None, reads from self.csv_path.
            upscaling_factor: The factor by which to upscale the grid.
            
        Returns:
            An upscaled grid.
        """
        if grid is None:
            grid = self.get_grid_data()
            
        upscale_factor = upscaling_factor * 2 - 1  # ensure odd number
        
        upscaled_grid = np.zeros(
            (len(grid) * upscale_factor, len(grid[0]) * upscale_factor)
        )
        
        for x, row in enumerate(grid):
            for y, cell in enumerate(row):
                if cell in [2, 3]:
                    upscaled_grid[x * upscale_factor + upscale_factor // 2, 
                                  y * upscale_factor + upscale_factor // 2] = cell
                else:
                    upscaled_grid[x * upscale_factor:(x + 1) * upscale_factor,
                                  y * upscale_factor:(y + 1) * upscale_factor] = cell
        
        return upscaled_grid
    
    def process_grid(self, csv_path=None, upscaling_factor=4):
        """
        Processes a grid by upscaling and padding it.
        
        Args:
            csv_path: Path to the CSV file. If None, uses self.csv_path.
            upscaling_factor: The factor by which to upscale the grid.
            
        Returns:
            A processed grid.
        """
        if csv_path is None:
            csv_path = self.csv_path
            
        starting_grid = self.get_grid_data(csv_path)
        
        upscale_factor = upscaling_factor * 2 - 1  # ensure odd number
        
        upscaled_grid = self.upscale_grid(starting_grid, upscaling_factor)
        
        padded_grid = self.get_padded_grid(upscaled_grid, radius=max(upscale_factor - 1, 1))
        
        return padded_grid


    def initialize_robot(self):
        print("SERIAL NUMBER:",self.robot_sn)
        self.ep_robot.initialize(conn_type="sta", sn=self.robot_sn)

        self.ep_robot.camera.start_video_stream(
            display=False, resolution=camera.STREAM_360P
        )

        self.ep_robot.robotic_arm.move(x=0, y=-100).wait_for_completed()

        self.ep_robot.gripper.open(power=70)

        time.sleep(1)
        self.ep_robot.gripper.pause()

        # Subscribe to the position
        self.ep_robot.chassis.sub_position(cs=0, freq=5, callback=self.position_callback)
        
        self.ep_robot.chassis.sub_attitude(freq=5, callback=self.attitude_callback)

        # Initialize grid data after robot is initialized
        self.grid = self.get_grid_data()
        self.processed_grid = self.process_grid()
        
        # Find robot starting position (marked by '2' in the grid)
        starting_pos = np.where(self.grid == self.starting_pos_number)
        if len(starting_pos[0]) > 0 and len(starting_pos[1]) > 0:
            self.start_grid_x = starting_pos[0][0]
            self.start_grid_y = starting_pos[1][0]
            print(f"Robot starting position found at grid cell ({self.start_grid_x}, {self.start_grid_y})")
        else:
            # Default to (0,0) if no starting position found
            self.start_grid_x = 0
            self.start_grid_y = 0
            print("Warning: No starting position (value 2) found in grid, using (0,0)")
        
        self.current_state = Project3States.LOOKING_FOR_BLOCK_IN_CLOSET

    def get_target_closet_position(self, closet_number):
        """Find the coordinates of a closet with the given number in the grid"""
        # Use np.where to find all coordinates where the grid value equals closet_number
        closet_coords = np.where(self.grid == closet_number)
        
        if len(closet_coords[0]) > 0 and len(closet_coords[1]) > 0:
            # Get the first matching position
            target_x, target_y = closet_coords[0][0], closet_coords[1][0]
            print(f"Found closet {closet_number} at grid position ({target_x}, {target_y})")
            
            # Convert grid coordinates to real-world coordinates
            real_x = target_x * self.cube_size_meters
            real_y = target_y * self.cube_size_meters
            return real_x, real_y
        else:
            print(f"Warning: Closet number {closet_number} not found in grid")
            return None, None

    def handle_looking_for_block_in_closet(self):

        # get the robot's current position
        current_position = self.get_position()

        # Get the current x, y, z from current_position
        current_x, current_y, current_z = current_position

        # Get the target x, y, z from self_closet_number
        target_x, target_y = self.get_target_closet_position(self.self_closet_number)

        # Calculate the angle between the robot and the target
        angle = np.arctan2(target_y - current_y, target_x - current_x)
        # dif_x = target_x - current_x
        # dif_y = target_y - current_y
        # # Rotate the robot to face the target
        #self.ep_robot.chassis.rotate(angle)
        frame = self.get_frame()
        detections, _ = self.run_yolo_detection(frame)
        found_object = self.get_target_label_detection(
            self.target_label, detections
        )
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray.astype(np.uint8)


        print("ANGLE:",angle)
        detections = self.apriltag.find_tags(gray)
        if detections:
            print("Detections",detections)

        self.ep_robot.chassis.move(x=0, y=0, z=np.rad2deg(angle))

        #self.ep_robot.chassis.move

    def run(self):
        self.initialize_robot()

        while self.current_state not in [
            Project3States.ERROR,
        ]:
            if cv2.waitKey(1) == ord("q"):
                break

            current_time_str = time.strftime("%Y-%m-%d %H:%M:%S")
            print(f"\n--- {current_time_str} | State: {self.current_state.value} ---")

            try:
                if self.current_state == Project3States.INITIALIZING:
                    self.initialize_robot()
                elif self.current_state == Project3States.LOOKING_FOR_BLOCK_IN_CLOSET:
                    self.handle_looking_for_block_in_closet()
                # elif self.current_state == Project3States.REMOVE_FROM_CLOSET:
                #     self.handle_remove_from_closet()
                # elif self.current_state == Project3States.LOOKING_FOR_BLOCK:
                #     self.handle_looking_for_block()
                # elif self.current_state == Project3States.APPROACH_BLOCK:
                #     self.handle_approach_block()
                # elif self.current_state == Project3States.GRAB_BLOCK:
                #     self.handle_grab_block()
                # elif self.current_state == Project3States.LIFT_ARM:
                #     self.handle_lift_arm()
                # elif self.current_state == Project3States.LOWER_ARM:
                #     self.handle_lower_arm()
                # elif self.current_state == Project3States.DROP_OFF:
                #     self.handle_drop_off()
                # elif self.current_state == Project3States.BACKUP:
                #     self.handle_backup()
                # elif self.current_state == Project3States.DELIVER_BLOCK:
                #     self.handle_deliver_block()
                # elif self.current_state == Project3States.BULLY_MODE:
                #     self.handle_bully_mode()
                # elif self.current_state == Project3States.WALL_MODE:
                #     self.handle_wall_mode()
            except Exception as e:
                print(traceback.format_exc())
                self.current_state = Project3States.ERROR
                self.ep_robot.chassis.unsub_position()
                self.ep_robot.chassis.unsub_attitude()
                break
            

        print(
            f"\n=== State Machine Finished with State: {self.current_state.value} ==="
        )
        
        # self.ep_robot.chassis._unsub_drone_all_status()
        self.reset_robot()  




if __name__ == "__main__":
    # More legible printing from numpy.
    

    state_machine = Project3StateMachine("3JKCH8800100YN", "./InitialMap.csv")

    try:
        state_machine.run()
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(traceback.format_exc())
    finally:
        print("Waiting for robomaster shutdown")


