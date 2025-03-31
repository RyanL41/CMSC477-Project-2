# Necessary Imports
import time
import cv2
import numpy as np
from ultralytics import YOLO
from robomaster import robot, camera, chassis, arm, gripper # Added arm, gripper
from enum import Enum
import math # For calculations if needed

# --- Constants ---
# Movement & Timing
ROTATION_SPEED = 20 # degrees per second
APPROACH_SPEED_X = 0.15 # m/s forward speed
APPROACH_SPEED_Z = 25 # degrees/s turning speed
BACKUP_DIST_10_FT = 3.048 # 10 feet in meters
BACKUP_DIST_5_FT = 1.524 # 5 feet in meters
MOVE_TIMEOUT = 10 # Max seconds for a move action (like backup)
SEARCH_TIMEOUT = 30 # Max seconds to rotate while searching
ARM_LIFT_Y = 80 # Vertical distance to lift arm (mm) - adjust as needed
ARM_LOWER_Y = -80 # Vertical distance to lower arm (mm) - adjust as needed
ARM_MOVE_SPEED = 30 # Speed for arm movement
GRIPPER_POWER = 50 # Power for gripper operation

# YOLO & Camera
YOLO_MODEL_PATH = "best.pt" # Path to your trained YOLO model
CONFIDENCE_THRESHOLD = 0.60 # Minimum confidence for YOLO detection
FRAME_WIDTH = 640 # Assuming 360p stream (check resolution if different)
FRAME_HEIGHT = 360
FRAME_CENTER_X = FRAME_WIDTH / 2
TARGET_BBOX_WIDTH_APPROACH = 150 # Target bbox width to stop approach (pixels) - TUNE THIS!
# --- Distance Calculation (Optional but potentially useful for approach) ---
# From: Z = object_height / (pixels / camera_focal_length)
# Rearranged: Z = (object_height * camera_focal_length) / pixels
# Requires knowing the real height of the object and camera focal length in pixels
OBJECT_REAL_HEIGHT_M = 0.05 # ASSUMPTION: Real height of blocks/targets in meters - TUNE THIS!
CAMERA_FOCAL_LENGTH_PIXELS = 314 # From AprilTag K matrix (fx, fy avg) - VERIFY THIS
TARGET_APPROACH_DISTANCE_M = 0.2 # Target distance to stop approach (meters) - TUNE THIS!

# --- State Definitions ---
class Project2States(Enum):
    INITIALIZING = "initializing"
    FIND_FIRST_BLOCK = "find_first_block"
    APPROACH_FIRST_BLOCK = "approach_first_block"
    GRAB_FIRST_BLOCK = "grab_first_block"
    LIFT_ARM_AFTER_GRAB1 = "lift_arm_after_grab1"
    BACKUP_AFTER_GRAB1 = "backup_after_grab1"
    RELEASE_FIRST_BLOCK_TEMP = "release_first_block_temp"
    LOWER_ARM_AFTER_RELEASE1 = "lower_arm_after_release1"
    BACKUP_AND_MOVE_CAM = "backup_and_move_cam" # Note: Camera position isn't directly controllable like gimbal
    SURVEY_FOR_BLOCK2 = "survey_for_block2"
    APPROACH_BLOCK2 = "approach_block2"
    GRAB_BLOCK2 = "grab_block2"
    SURVEY_FOR_TARGET1 = "survey_for_target1"
    APPROACH_TARGET1 = "approach_target1"
    RELEASE_BLOCK2_AT_TARGET1 = "release_block2_at_target1"
    LOWER_ARM_AFTER_RELEASE2 = "lower_arm_after_release2" # Added lower arm state
    BACKUP_AFTER_TARGET1 = "backup_after_target1" # Added backup state
    SURVEY_FOR_BLOCK1_AGAIN = "survey_for_block1_again"
    APPROACH_BLOCK1_AGAIN = "approach_block1_again"
    GRAB_BLOCK1_AGAIN = "grab_block1_again"
    LIFT_ARM_AFTER_GRAB2 = "lift_arm_after_grab2" # Added lift arm state
    SURVEY_FOR_TARGET2 = "survey_for_target2"
    APPROACH_TARGET2 = "approach_target2"
    RELEASE_BLOCK1_AT_TARGET2 = "release_block1_at_target2"
    LOWER_ARM_AFTER_RELEASE3 = "lower_arm_after_release3" # Added lower arm state
    COMPLETED = "completed"
    ERROR = "error"

hi = 1

class Project2StateMachine:
    def __init__(self, robot_sn):
        self.robot_sn = robot_sn
        self.ep_robot = None
        self.ep_camera = None
        self.ep_chassis = None
        self.ep_arm = None
        self.ep_gripper = None
        self.yolo_model = None

        self.current_state = Project2States.INITIALIZING
        self.target_label = None # Label of the object we are currently looking for/interacting with
        self.last_detection = None # Store details of the last relevant detection (box, label, conf)
        self.start_time = time.time() # For timeouts

    def initialize_robot(self):
        """Initializes connection to the robot"""
        print("Initializing Robot...")
        try:
            self.ep_robot = robot.Robot()
            self.ep_robot.initialize(conn_type="sta", sn=self.robot_sn)
            self.ep_camera = self.ep_robot.camera
            self.ep_chassis = self.ep_robot.chassis
            self.ep_arm = self.ep_robot.robotic_arm # Correct object for arm control
            self.ep_gripper = self.ep_robot.gripper # Correct object for gripper control

            # Initialize camera stream
            self.ep_camera.start_video_stream(display=False, resolution=camera.STREAM_360P)
            print("Robot Camera Initialized.")

            # Load YOLO Model
            print(f"Loading YOLO model from {YOLO_MODEL_PATH}...")
            self.yolo_model = YOLO(YOLO_MODEL_PATH)
            print("YOLO Model Loaded.")

            # Set initial arm/gripper state (optional, but good practice)
            print("Setting initial gripper/arm state...")
            # Center arm? Robomaster arm starts centered.
            # self.ep_arm.moveto(x=200, y=0).wait_for_completed() # Example: Move arm to a known start pos
            self.ep_gripper.open(power=GRIPPER_POWER)
            time.sleep(1)
            self.ep_gripper.pause()
            print("Initial states set.")

            return True
        except Exception as e:
            print(f"Error during robot initialization: {e}")
            self.current_state = Project2States.ERROR
            return False

    def cleanup(self):
        """Cleans up resources"""
        print("Cleaning up...")
        if self.ep_camera:
            self.ep_camera.stop_video_stream()
        if self.ep_robot and self.ep_robot.is_connected:
             # Ensure robot stops moving
            if self.ep_chassis:
                self.ep_chassis.drive_speed(x=0, y=0, z=0)
            # Ensure gripper is paused
            if self.ep_gripper:
                self.ep_gripper.pause()
            self.ep_robot.close()
        print("Cleanup complete.")

    def get_frame(self):
        """Reads the latest frame from the camera"""
        try:
            frame = self.ep_camera.read_cv2_image(strategy="newest", timeout=1.0)
            if frame is None:
                print("Warning: Failed to get frame.")
            return frame
        except Exception as e:
            print(f"Error reading frame: {e}")
            return None

    def run_yolo_detection(self, frame):
        """Runs YOLO detection on a frame and returns results"""
        if frame is None or self.yolo_model is None:
            return None, None # Return None for results and frame if input is bad

        results = self.yolo_model.predict(source=frame, show=False, verbose=False)[0] # Get first result, disable showing/verbosity

        # --- Simple Visualization (Optional) ---
        boxes = results.boxes
        class_names = self.yolo_model.names
        vis_frame = frame.copy() # Draw on a copy

        detections_list = []

        for box in boxes:
            xyxy = box.xyxy.cpu().numpy().flatten()
            class_id = int(box.cls.cpu().numpy())
            label = class_names[class_id]
            confidence = float(box.conf.cpu().numpy()) # Convert to float

            if confidence >= CONFIDENCE_THRESHOLD:
                 detections_list.append({
                     'label': label,
                     'confidence': confidence,
                     'box': xyxy # [x1, y1, x2, y2]
                 })
                 # Draw bounding box
                 cv2.rectangle(vis_frame,
                               (int(xyxy[0]), int(xyxy[1])),
                               (int(xyxy[2]), int(xyxy[3])),
                               color=(0, 255, 0), thickness=2)
                 # Add label and confidence
                 label_text = f"{label} ({confidence:.2f})"
                 cv2.putText(vis_frame, label_text, (int(xyxy[0]), int(xyxy[1]) - 10),
                             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # Display the frame (optional for debugging)
        # cv2.imshow('YOLO Detections', vis_frame)
        # cv2.waitKey(1) # Necessary for imshow to work

        return detections_list, vis_frame


    def find_object_with_yolo(self, target_label, detections):
        """Searches detections list for a specific label"""
        if not detections:
            return None

        best_detection = None
        max_confidence = 0

        for det in detections:
            if det['label'] == target_label and det['confidence'] > max_confidence:
                 max_confidence = det['confidence']
                 best_detection = det

        return best_detection # Returns the highest confidence detection matching the label


    def calculate_distance_from_bbox_height(self, bbox_height_pixels):
        """Estimates distance using bbox height. Returns distance in meters."""
        if bbox_height_pixels <= 0:
            return float('inf') # Avoid division by zero, return infinity

        # Z = (object_height * camera_focal_length) / pixels
        distance = (OBJECT_REAL_HEIGHT_M * CAMERA_FOCAL_LENGTH_PIXELS) / bbox_height_pixels
        return distance

    def approach_object_simple(self, detection):
        """Calculates chassis velocity to approach the detected object based on bbox"""
        if not detection:
            return 0, 0, 0 # No movement if no detection

        x1, y1, x2, y2 = detection['box']
        box_center_x = (x1 + x2) / 2
        # box_center_y = (y1 + y2) / 2 # Not typically used for driving
        box_width = x2 - x1
        # box_height = y2 - y1 # Use for distance if needed

        # --- Calculate distance (optional, using simpler width threshold here) ---
        # distance = self.calculate_distance_from_bbox_height(box_height)
        # print(f"Estimated distance: {distance:.2f} m")
        # is_close_enough = distance < TARGET_APPROACH_DISTANCE_M

        # --- Using bbox width as proxy for closeness ---
        is_close_enough = box_width > TARGET_BBOX_WIDTH_APPROACH
        #print(f"Box Width: {box_width:.1f}, Target Width: {TARGET_BBOX_WIDTH_APPROACH}")


        if is_close_enough:
            print("Close enough to object.")
            return 0, 0, 0 # Stop moving

        # --- Proportional Control ---
        # Calculate error (difference from center)
        error_x = FRAME_CENTER_X - box_center_x

        # Calculate velocities
        # Forward speed (constant when not close)
        x_vel = APPROACH_SPEED_X

        # Turning speed (proportional to how far off-center the object is)
        # Adjust the scaling factor (e.g., 0.1) to control turn sensitivity
        z_vel = np.clip(error_x * 0.1, -APPROACH_SPEED_Z, APPROACH_SPEED_Z)

        # y_vel is usually 0 for forward/turning movement
        y_vel = 0

        # print(f"Approach Vels: x={x_vel:.2f}, z={z_vel:.2f}")
        return x_vel, y_vel, z_vel


    # --- State Handling Methods ---

    def handle_initializing(self):
        if self.initialize_robot():
            print("Initialization successful. Moving to FIND_FIRST_BLOCK.")
            self.current_state = Project2States.FIND_FIRST_BLOCK
        else:
            print("Initialization failed. Entering ERROR state.")
            self.current_state = Project2States.ERROR

    def handle_find_object(self, next_state_on_find, search_label):
        """Generic handler for rotating and searching"""
        print(f"State: Searching for {search_label}...")
        self.target_label = search_label
        self.ep_chassis.drive_speed(x=0, y=0, z=ROTATION_SPEED) # Start rotating
        self.start_time = time.time() # Reset timeout timer

        while time.time() - self.start_time < SEARCH_TIMEOUT:
            frame = self.get_frame()
            detections, _ = self.run_yolo_detection(frame)
            found_object = self.find_object_with_yolo(self.target_label, detections)

            if found_object:
                print(f"Found {self.target_label}!")
                self.ep_chassis.drive_speed(x=0, y=0, z=0) # Stop rotating
                self.last_detection = found_object
                self.current_state = next_state_on_find
                return # Exit the handler

            if cv2.waitKey(1) & 0xFF == ord('q'): # Allow manual quit
                 self.current_state = Project2States.ERROR
                 return

            time.sleep(0.05) # Small delay

        # Timeout occurred
        print(f"Timeout: Could not find {self.target_label} within {SEARCH_TIMEOUT}s.")
        self.ep_chassis.drive_speed(x=0, y=0, z=0)
        self.current_state = Project2States.ERROR


    def handle_approach_object(self, next_state_on_approach, target_label):
         """Generic handler for approaching a detected object"""
         print(f"State: Approaching {target_label}...")
         self.target_label = target_label # Ensure target label is set

         frame = self.get_frame()
         detections, _ = self.run_yolo_detection(frame)
         # Try to find the specific object we were tracking
         current_detection = self.find_object_with_yolo(self.target_label, detections)

         if not current_detection:
            # Lost the object, maybe go back to searching? Or just stop?
            print(f"Lost sight of {self.target_label} while approaching.")
            # Option 1: Stop and maybe re-search later
            self.ep_chassis.drive_speed(x=0, y=0, z=0)
            # Maybe transition back to a find state or error
            # For now, let's try to proceed based on last known good detection if available
            if not self.last_detection or self.last_detection['label'] != self.target_label:
                 print("No valid last detection. Going to ERROR.")
                 self.current_state = Project2States.ERROR
                 return
            else:
                 print("Using last known detection to estimate approach.")
                 current_detection = self.last_detection # Fallback

         # Calculate movement based on the current (or last known) detection
         x_vel, y_vel, z_vel = self.approach_object_simple(current_detection)

         # Update last detection if we have a fresh one
         if current_detection and current_detection['label'] == self.target_label:
             self.last_detection = current_detection

         if x_vel == 0 and y_vel == 0 and z_vel == 0:
             # Approach calculation determined we are close enough
             print(f"Approach complete for {self.target_label}.")
             self.ep_chassis.drive_speed(x=0, y=0, z=0) # Make sure robot is stopped
             self.current_state = next_state_on_approach
         else:
             # Continue approaching
             self.ep_chassis.drive_speed(x=x_vel, y=y_vel, z=z_vel)

         # Add a small delay or check frequency control
         time.sleep(0.1)


    def handle_grab(self, next_state):
        print(f"State: Grabbing...")
        try:
            self.ep_gripper.close(power=GRIPPER_POWER)
            time.sleep(1.5) # Allow time for gripper to close
            self.ep_gripper.pause()
            print("Grab complete.")
            self.current_state = next_state
        except Exception as e:
            print(f"Error during grab: {e}")
            self.current_state = Project2States.ERROR

    def handle_release(self, next_state):
        print(f"State: Releasing...")
        try:
            self.ep_gripper.open(power=GRIPPER_POWER)
            time.sleep(1.5) # Allow time for gripper to open
            self.ep_gripper.pause()
            print("Release complete.")
            self.current_state = next_state
        except Exception as e:
            print(f"Error during release: {e}")
            self.current_state = Project2States.ERROR

    def handle_move_arm(self, y_distance, next_state):
        """Handles moving the arm vertically"""
        print(f"State: Moving arm y={y_distance}mm...")
        try:
            # Use moveto for absolute or move for relative - move seems more appropriate here
            # wait_for_completed makes it blocking
            self.ep_arm.move(y=y_distance, y_speed=ARM_MOVE_SPEED).wait_for_completed()
            print("Arm move complete.")
            self.current_state = next_state
        except Exception as e:
            print(f"Error moving arm: {e}")
            self.current_state = Project2States.ERROR

    def handle_backup(self, distance_m, next_state):
        """Handles moving the chassis backward"""
        print(f"State: Backing up {distance_m}m...")
        try:
            # Use chassis.move for controlled distance movement (blocking)
            self.ep_chassis.move(x=-distance_m, y=0, z=0, xy_speed=APPROACH_SPEED_X).wait_for_completed()
            print("Backup complete.")
            self.current_state = next_state
        except Exception as e:
            print(f"Error during backup: {e}")
            self.current_state = Project2States.ERROR


    # --- Main Loop ---
    def run(self):
        if not self.initialize_robot():
             self.cleanup()
             return

        while self.current_state != Project2States.COMPLETED and self.current_state != Project2States.ERROR:
            current_time_str = time.strftime("%Y-%m-%d %H:%M:%S")
            # print(f"--- {current_time_str} | State: {self.current_state.value} ---") # Optional verbose logging

            try:
                # --- State transitions based on the defined sequence ---
                if self.current_state == Project2States.INITIALIZING:
                     self.handle_initializing() # Should transition state internally

                elif self.current_state == Project2States.FIND_FIRST_BLOCK:
                    self.handle_find_object(Project2States.APPROACH_FIRST_BLOCK, "Block 1")

                elif self.current_state == Project2States.APPROACH_FIRST_BLOCK:
                    self.handle_approach_object(Project2States.GRAB_FIRST_BLOCK, "Block 1")

                elif self.current_state == Project2States.GRAB_FIRST_BLOCK:
                    self.handle_grab(Project2States.LIFT_ARM_AFTER_GRAB1)

                elif self.current_state == Project2States.LIFT_ARM_AFTER_GRAB1:
                    self.handle_move_arm(ARM_LIFT_Y, Project2States.BACKUP_AFTER_GRAB1)

                elif self.current_state == Project2States.BACKUP_AFTER_GRAB1:
                    self.handle_backup(BACKUP_DIST_10_FT, Project2States.RELEASE_FIRST_BLOCK_TEMP)

                elif self.current_state == Project2States.RELEASE_FIRST_BLOCK_TEMP:
                    self.handle_release(Project2States.LOWER_ARM_AFTER_RELEASE1)

                elif self.current_state == Project2States.LOWER_ARM_AFTER_RELEASE1:
                     self.handle_move_arm(ARM_LOWER_Y, Project2States.BACKUP_AND_MOVE_CAM)

                elif self.current_state == Project2States.BACKUP_AND_MOVE_CAM:
                     # "Move camera up to center" - Arm is likely already centered after lowering.
                     # The main action here is backing up. Gripper state is open from previous step.
                     self.handle_backup(BACKUP_DIST_5_FT, Project2States.SURVEY_FOR_BLOCK2)

                elif self.current_state == Project2States.SURVEY_FOR_BLOCK2:
                     self.handle_find_object(Project2States.APPROACH_BLOCK2, "Block 2")

                elif self.current_state == Project2States.APPROACH_BLOCK2:
                     self.handle_approach_object(Project2States.GRAB_BLOCK2, "Block 2")

                elif self.current_state == Project2States.GRAB_BLOCK2:
                     self.handle_grab(Project2States.SURVEY_FOR_TARGET1) # Lift arm not specified here

                elif self.current_state == Project2States.SURVEY_FOR_TARGET1:
                     self.handle_find_object(Project2States.APPROACH_TARGET1, "Target 1")

                elif self.current_state == Project2States.APPROACH_TARGET1:
                     self.handle_approach_object(Project2States.RELEASE_BLOCK2_AT_TARGET1, "Target 1")

                elif self.current_state == Project2States.RELEASE_BLOCK2_AT_TARGET1:
                     self.handle_release(Project2States.LOWER_ARM_AFTER_RELEASE2) # Go to lower arm

                elif self.current_state == Project2States.LOWER_ARM_AFTER_RELEASE2:
                    self.handle_move_arm(ARM_LOWER_Y, Project2States.BACKUP_AFTER_TARGET1)

                elif self.current_state == Project2States.BACKUP_AFTER_TARGET1:
                    self.handle_backup(0.5, Project2States.SURVEY_FOR_BLOCK1_AGAIN) # Small backup

                elif self.current_state == Project2States.SURVEY_FOR_BLOCK1_AGAIN:
                     self.handle_find_object(Project2States.APPROACH_BLOCK1_AGAIN, "Block 1")

                elif self.current_state == Project2States.APPROACH_BLOCK1_AGAIN:
                     self.handle_approach_object(Project2States.GRAB_BLOCK1_AGAIN, "Block 1")

                elif self.current_state == Project2States.GRAB_BLOCK1_AGAIN:
                     self.handle_grab(Project2States.LIFT_ARM_AFTER_GRAB2) # Lift after grab

                elif self.current_state == Project2States.LIFT_ARM_AFTER_GRAB2:
                    self.handle_move_arm(ARM_LIFT_Y, Project2States.SURVEY_FOR_TARGET2)

                elif self.current_state == Project2States.SURVEY_FOR_TARGET2:
                     self.handle_find_object(Project2States.APPROACH_TARGET2, "Target 2")

                elif self.current_state == Project2States.APPROACH_TARGET2:
                     self.handle_approach_object(Project2States.RELEASE_BLOCK1_AT_TARGET2, "Target 2")

                elif self.current_state == Project2States.RELEASE_BLOCK1_AT_TARGET2:
                     # "Drop Block" - assumes release is sufficient
                     self.handle_release(Project2States.LOWER_ARM_AFTER_RELEASE3) # Lower arm

                elif self.current_state == Project2States.LOWER_ARM_AFTER_RELEASE3:
                     self.handle_move_arm(ARM_LOWER_Y, Project2States.COMPLETED) # Go to completed

                else:
                    print(f"Unknown state: {self.current_state}. Stopping.")
                    self.current_state = Project2States.ERROR

                # Small delay to prevent high CPU usage and allow hardware commands
                time.sleep(0.01)

                # Allow quitting by pressing 'q' in the OpenCV window if shown
                if cv2.waitKey(1) & 0xFF == ord('q'):
                     print("Manual quit requested.")
                     self.current_state = Project2States.ERROR
                     break

            except KeyboardInterrupt:
                print("Keyboard interrupt detected. Stopping.")
                self.current_state = Project2States.ERROR
                break
            except Exception as e:
                print(f"An unexpected error occurred in state {self.current_state}: {e}")
                import traceback
                traceback.print_exc()
                self.current_state = Project2States.ERROR
                break

        # End of loop
        if self.current_state == Project2States.COMPLETED:
            print("--- Project 2 Sequence Completed Successfully! ---")
        else:
            print("--- Project 2 Sequence Ended with Error or Interruption ---")

        self.cleanup()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    ROBOT_SN = "3JKCH8800100YN" # Replace with your robot's Serial Number

    state_machine = Project2StateMachine(robot_sn=ROBOT_SN)
    state_machine.run()