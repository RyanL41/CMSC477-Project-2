# Necessary Imports
import time
import cv2
import numpy as np
from ultralytics import YOLO
from robomaster import robot, camera, chassis # Explicit imports
from enum import Enum
import math
import traceback # For detailed error reporting

# --- Constants ---
# Movement & Timing
ROTATION_SPEED = 20         # degrees per second for searching
APPROACH_SPEED_X = 0.15     # m/s forward speed during approach
APPROACH_SPEED_Z = 25       # degrees/s turning speed during approach
BACKUP_DIST_LONG = 3.048    # meters (approx 10 ft)
BACKUP_DIST_SHORT = 1.524   # meters (approx 5 ft)
BACKUP_DIST_VERY_SHORT = 0.5 # meters
MOVE_TIMEOUT = 15           # Max seconds for a chassis move action
SEARCH_TIMEOUT = 30         # Max seconds to rotate while searching
ARM_LIFT_Y = 80             # Vertical distance to lift arm (mm)
ARM_LOWER_Y = -80           # Vertical distance to lower arm (mm)
ARM_MOVE_SPEED = 30         # Speed for arm movement (mm/s)
GRIPPER_POWER = 50          # Power for gripper operation

# YOLO & Camera
YOLO_MODEL_PATH = "best.pt" # Path to your trained YOLO model
CONFIDENCE_THRESHOLD = 0.60 # Minimum confidence for YOLO detection
FRAME_WIDTH = 640           # Camera stream width
FRAME_HEIGHT = 360          # Camera stream height
FRAME_CENTER_X = FRAME_WIDTH / 2
# Target bbox width in pixels to determine closeness during approach - TUNE THIS!
TARGET_BBOX_WIDTH_APPROACH = 150

# --- Optional Distance Calculation Parameters (Requires Tuning/Calibration) ---
OBJECT_REAL_HEIGHT_M = 0.05 # ASSUMPTION: Real height of blocks/targets in meters
CAMERA_FOCAL_LENGTH_PIXELS = 314 # From camera intrinsics (fx, fy avg) - VERIFY THIS
TARGET_APPROACH_DISTANCE_M = 0.2 # Target distance to stop approach (meters)

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
    BACKUP_AND_RESET_ARM = "backup_and_reset_arm" # Renamed for clarity
    SURVEY_FOR_BLOCK2 = "survey_for_block2"
    APPROACH_BLOCK2 = "approach_block2"
    GRAB_BLOCK2 = "grab_block2"
    LIFT_ARM_AFTER_GRAB2 = "lift_arm_after_grab2" # Lift after grab 2
    SURVEY_FOR_TARGET1 = "survey_for_target1"
    APPROACH_TARGET1 = "approach_target1"
    RELEASE_BLOCK2_AT_TARGET1 = "release_block2_at_target1"
    LOWER_ARM_AFTER_RELEASE2 = "lower_arm_after_release2"
    BACKUP_AFTER_TARGET1 = "backup_after_target1"
    SURVEY_FOR_BLOCK1_AGAIN = "survey_for_block1_again"
    APPROACH_BLOCK1_AGAIN = "approach_block1_again"
    GRAB_BLOCK1_AGAIN = "grab_block1_again"
    LIFT_ARM_AFTER_GRAB3 = "lift_arm_after_grab3" # Lift after grab 1 (again)
    SURVEY_FOR_TARGET2 = "survey_for_target2"
    APPROACH_TARGET2 = "approach_target2"
    RELEASE_BLOCK1_AT_TARGET2 = "release_block1_at_target2"
    LOWER_ARM_AFTER_RELEASE3 = "lower_arm_after_release3"
    COMPLETED = "completed"
    ERROR = "error"

class Project2StateMachine:
    def __init__(self, robot_sn):
        self.robot_sn = robot_sn
        self.ep_robot: robot.Robot = None
        self.ep_camera: camera.Camera = None
        self.ep_chassis: chassis.Chassis = None
        self.ep_arm =  None
        self.ep_gripper = None
        self.yolo_model: YOLO = None

        self.current_state = Project2States.INITIALIZING
        self.target_label = None # Label of the object we are currently looking for/interacting with
        self.last_detection = None # Store details of the last relevant detection (box, label, conf)
        self.start_time = time.time() # For timeouts

        # For optional live view
        self.show_video = True
        self.last_vis_frame = None

    def initialize_robot(self):
        """Initializes connection to the robot and its components."""
        print("Initializing Robot...")
        try:
            self.ep_robot = robot.Robot()
            # Use Robot SN if provided, otherwise discover
            init_args = {"conn_type": "sta"}
            if self.robot_sn:
                init_args["sn"] = self.robot_sn
            self.ep_robot.initialize(**init_args)

            self.ep_camera = self.ep_robot.camera
            self.ep_chassis = self.ep_robot.chassis
            self.ep_arm = self.ep_robot.robotic_arm
            self.ep_gripper = self.ep_robot.gripper

            # Initialize camera stream
            self.ep_camera.start_video_stream(display=False, resolution=camera.STREAM_360P)
            print("Robot Camera Initialized.")

            # Load YOLO Model
            print(f"Loading YOLO model from {YOLO_MODEL_PATH}...")
            self.yolo_model = YOLO(YOLO_MODEL_PATH)
            # Disable verbose output during prediction
            if hasattr(self.yolo_model, 'predictor') and self.yolo_model.predictor:
                 self.yolo_model.predictor.args.verbose = False
            print("YOLO Model Loaded.")

            # Set initial arm/gripper state
            print("Setting initial gripper/arm state...")
            # self.ep_arm.moveto(x=180, y=0).wait_for_completed() # Example: Move arm to a known neutral start pos
            self.ep_gripper.open(power=GRIPPER_POWER)
            time.sleep(1) # Allow time for opening
            self.ep_gripper.pause() # Conserve power
            print("Initial states set.")

            return True
        except Exception as e:
            print(f"ERROR during robot initialization: {e}")
            self.current_state = Project2States.ERROR
            return False

    def cleanup(self):
        """Cleans up resources (camera, robot connection)."""
        print("Cleaning up...")
        if self.show_video:
            cv2.destroyAllWindows()
        if self.ep_camera:
            self.ep_camera.stop_video_stream()
        if self.ep_robot:
             # Ensure robot stops moving
            if self.ep_chassis:
                self.ep_chassis.drive_speed(x=0, y=0, z=0)
            # Ensure gripper is paused
            if self.ep_gripper:
                self.ep_gripper.pause()
            self.ep_robot.close()
        print("Cleanup complete.")

    def get_frame(self):
        """Reads the latest frame from the camera."""
        try:
            # Use strategy="newest" to get the most recent frame, reducing lag
            frame = self.ep_camera.read_cv2_image(strategy="newest", timeout=1.0)
            if frame is None:
                print("Warning: Failed to get frame.")
                time.sleep(0.1) # Avoid spamming warnings if connection flickers
            return frame
        except Exception as e:
            print(f"ERROR reading frame: {e}")
            time.sleep(0.5) # Pause briefly after error
            return None

    def run_yolo_detection(self, frame):
        """Runs YOLO detection on a frame, logs detections, and returns results and visualized frame."""
        if frame is None or self.yolo_model is None:
            return [], None # Return empty list and None frame

        # --- Run Prediction ---
        # Disable Ultralytics' default plotting and verbose output for cleaner logs/performance
        results = self.yolo_model.predict(source=frame, show=False, verbose=False, conf=CONFIDENCE_THRESHOLD)[0]

        # --- Process & Log Detections ---
        boxes = results.boxes
        class_names = self.yolo_model.names
        vis_frame = frame.copy() # Draw on a copy
        detections_list = []

        # print(f"  [YOLO] Found {len(boxes)} potential boxes.") # Log raw box count
        for box in boxes:
            xyxy = box.xyxy.cpu().numpy().flatten().astype(int) # [x1, y1, x2, y2]
            class_id = int(box.cls.cpu().numpy())
            label = class_names.get(class_id, f"Unknown({class_id})") # Use .get for safety
            confidence = float(box.conf.cpu().numpy())

            # Log detected boxes above threshold BEFORE adding to list
            print(f"    [Detection] Label: '{label}' (ID:{class_id}), Conf: {confidence:.2f}, Box: [{xyxy[0]} {xyxy[1]} {xyxy[2]} {xyxy[3]}]")

            detections_list.append({
                'label': label,
                'confidence': confidence,
                'box': xyxy
            })

            # --- Draw on vis_frame (Optional) ---
            if self.show_video:
                cv2.rectangle(vis_frame, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), color=(0, 255, 0), thickness=2)
                label_text = f"{label} ({confidence:.2f})"
                cv2.putText(vis_frame, label_text, (xyxy[0], xyxy[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        self.last_vis_frame = vis_frame # Store for potential display later
        return detections_list, vis_frame

    def find_object_with_yolo(self, target_label, detections):
        """Searches detections list for the highest confidence instance of a specific label."""
        if not detections:
            return None

        best_detection = None
        max_confidence = 0.0 # Initialize with 0.0

        for det in detections:
            if det['label'] == target_label and det['confidence'] > max_confidence:
                 max_confidence = det['confidence']
                 best_detection = det

        # if best_detection:
        #      print(f"    [Find] Best match for '{target_label}': Conf={max_confidence:.2f}")
        # else:
        #      print(f"    [Find] No match found for '{target_label}' in current detections.")

        return best_detection

    def calculate_distance_from_bbox_height(self, bbox_height_pixels):
        """Estimates distance using bbox height. Requires calibration."""
        if bbox_height_pixels <= 0:
            return float('inf') # Avoid division by zero
        # Z = (object_real_height_m * camera_focal_length_pixels) / bbox_height_pixels
        distance = (OBJECT_REAL_HEIGHT_M * CAMERA_FOCAL_LENGTH_PIXELS) / bbox_height_pixels
        return distance

    def approach_object_simple(self, detection):
        """Calculates chassis velocity (x, y, z) to approach the detected object based on bbox."""
        x_vel, y_vel, z_vel = 0.0, 0.0, 0.0 # Default to stop

        if not detection:
            print("  [Approach] No detection provided. Setting velocities to 0.")
            return x_vel, y_vel, z_vel # No movement if no detection

        x1, y1, x2, y2 = detection['box']
        print("X1:",x1,"X2:",x2)
        box_center_x = (x1 + x2) / 2
        box_width = x2 - x1
        # box_height = y2 - y1 # Could be used for distance estimation

        # --- Using bbox width as a proxy for closeness ---
        is_close_enough = box_width > TARGET_BBOX_WIDTH_APPROACH
        print(f"  [Approach] Box Center X: {box_center_x:.1f}, Width: {box_width:.1f} (Target > {TARGET_BBOX_WIDTH_APPROACH})")

        if is_close_enough:
            print("  [Approach] Close enough to object based on width. Setting velocities to 0.")
            return x_vel, y_vel, z_vel # Stop moving

        # --- Proportional Control for Alignment ---
        error_x = FRAME_CENTER_X - box_center_x
        # Scale turning speed based on error. Adjust the 0.1 factor to tune sensitivity.
        # Clamp speed to max approach turning speed.
        z_vel = np.clip(error_x * 0.1, -APPROACH_SPEED_Z, APPROACH_SPEED_Z)

        # --- Constant Forward Speed ---
        x_vel = APPROACH_SPEED_X

        # y_vel is typically 0 for ground robots unless strafing
        y_vel = 0.0

        print(f"  [Approach] Error X: {error_x:.1f}. Calculated Velocities: x={x_vel:.2f}, y={y_vel:.2f}, z={z_vel:.2f}")
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
        """Generic handler for rotating and searching for a specific object label."""
        print(f"State: Searching for '{search_label}'...")
        self.target_label = search_label
        self.ep_chassis.drive_speed(x=0, y=0, z=ROTATION_SPEED) # Start rotating
        self.start_time = time.time() # Reset timeout timer

        while time.time() - self.start_time < SEARCH_TIMEOUT:
            frame = self.get_frame()
            detections, _ = self.run_yolo_detection(frame) # Get detections, ignore vis_frame here
            found_object = self.find_object_with_yolo(self.target_label, detections)

            if found_object:
                print(f"FOUND '{self.target_label}'!")
                self.ep_chassis.drive_speed(x=0, y=0, z=0) # Stop rotating
                self.last_detection = found_object # Store the successful detection
                self.current_state = next_state_on_find
                return # Exit the handler

            if self.show_video and self.last_vis_frame is not None:
                cv2.imshow('Robot View', self.last_vis_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("Manual quit requested during search.")
                    self.current_state = Project2States.ERROR
                    return

            # time.sleep(0.02) # Small delay to prevent busy-waiting

        # Timeout occurred
        print(f"TIMEOUT: Could not find '{self.target_label}' within {SEARCH_TIMEOUT}s.")
        self.ep_chassis.drive_speed(x=0, y=0, z=0) # Ensure stopped
        self.current_state = Project2States.ERROR

    def handle_approach_object(self, next_state_on_approach, target_label):
         """Generic handler for approaching a detected object using simple proportional control."""
         print(f"State: Approaching '{target_label}'...")
         self.target_label = target_label # Ensure target label is set for clarity

         frame = self.get_frame()
         detections, _ = self.run_yolo_detection(frame)
         # Try to find the specific object we are tracking
         current_detection = self.find_object_with_yolo(self.target_label, detections)

         approach_detection = None
         if current_detection:
             # Use the fresh detection if available
             approach_detection = current_detection
             self.last_detection = current_detection # Update last known good detection
         elif self.last_detection and self.last_detection['label'] == self.target_label:
             # Fallback to the last known detection if we lost sight momentarily
             print(f"  [Approach] Lost sight of '{self.target_label}', using last known detection info.")
             approach_detection = self.last_detection
         else:
             # Lost the object and no valid last detection
             print(f"ERROR: Lost sight of '{self.target_label}' and no valid last detection. Cannot approach.")
             self.ep_chassis.drive_speed(x=0, y=0, z=0) # Stop
             self.current_state = Project2States.ERROR # Transition to error or maybe back to search?
             return

         # Calculate movement based on the selected detection
         x_vel, y_vel, z_vel = self.approach_object_simple(approach_detection)

         # Check if the approach calculation determined we are close enough (returned 0s)
         if x_vel == 0 and y_vel == 0 and z_vel == 0:
             print(f"  [Approach] Approach complete for '{self.target_label}'.")
             self.ep_chassis.drive_speed(x=0, y=0, z=0) # Make sure robot is stopped
             self.current_state = next_state_on_approach
         else:
             # Continue approaching
             self.ep_chassis.drive_speed(x=x_vel, y=y_vel, z=z_vel)

         # Update visualization if enabled
         if self.show_video and self.last_vis_frame is not None:
             cv2.imshow('Robot View', self.last_vis_frame)
             if cv2.waitKey(1) & 0xFF == ord('q'):
                 print("Manual quit requested during approach.")
                 self.current_state = Project2States.ERROR


    def handle_grab(self, next_state):
        """Handles closing the gripper."""
        print(f"State: Grabbing...")
        try:
            self.ep_gripper.close(power=GRIPPER_POWER)
            time.sleep(1.5) # Allow time for gripper to close physically
            # Check gripper status (optional, might not be perfectly reliable)
            # status = self.ep_gripper.get_status() # Can return OPEN, CLOSED, or UNKNOWN
            # print(f"  [Grab] Gripper status check: {status}")
            self.ep_gripper.pause() # Pause motor to save power
            print("  [Grab] Grab action complete.")
            self.current_state = next_state
        except Exception as e:
            print(f"ERROR during grab: {e}")
            self.current_state = Project2States.ERROR

    def handle_release(self, next_state):
        """Handles opening the gripper."""
        print(f"State: Releasing...")
        try:
            self.ep_gripper.open(power=GRIPPER_POWER)
            time.sleep(1.5) # Allow time for gripper to open physically
            self.ep_gripper.pause() # Pause motor
            print("  [Release] Release action complete.")
            self.current_state = next_state
        except Exception as e:
            print(f"ERROR during release: {e}")
            self.current_state = Project2States.ERROR

    def handle_move_arm(self, y_distance, next_state):
        """Handles moving the arm vertically by a relative distance (positive=up, negative=down)."""
        action = "Lifting" if y_distance > 0 else "Lowering"
        print(f"State: {action} arm y={y_distance}mm...")
        try:
            # Use relative move `arm.move()`
            # wait_for_completed makes it blocking until the move finishes
            self.ep_arm.move(y=y_distance, y_speed=ARM_MOVE_SPEED).wait_for_completed(timeout=10)
            print(f"  [Arm] Arm {action.lower()} complete.")
            self.current_state = next_state
        except Exception as e:
            print(f"ERROR moving arm: {e}")
            self.current_state = Project2States.ERROR

    def handle_backup(self, distance_m, next_state):
        """Handles moving the chassis backward by a specified distance."""
        print(f"State: Backing up {distance_m:.2f}m...")
        if distance_m <= 0: # Skip if distance is zero or negative
             print("  [Backup] Distance is zero, skipping move.")
             self.current_state = next_state
             return
        try:
            # Use chassis.move for controlled distance movement (blocking)
            # Note: x is forward/backward (+/-), y is left/right (+/-)
            self.ep_chassis.move(x=-distance_m, y=0, z=0, xy_speed=APPROACH_SPEED_X).wait_for_completed(timeout=MOVE_TIMEOUT)
            print("  [Backup] Backup move complete.")
            self.current_state = next_state
        except Exception as e:
            print(f"ERROR during backup: {e}")
             # Attempt to stop chassis just in case
            try:
                self.ep_chassis.drive_speed(x=0, y=0, z=0, timeout=1)
            except: pass
            self.current_state = Project2States.ERROR

    # --- Main Loop ---
    def run(self):
        """Runs the main state machine loop."""
        # Don't start if initialization fails immediately
        if self.current_state == Project2States.INITIALIZING:
            self.handle_initializing()
            if self.current_state == Project2States.ERROR:
                self.cleanup()
                return

        # Main operational loop
        while self.current_state not in [Project2States.COMPLETED, Project2States.ERROR]:
            start_loop_time = time.time()
            current_time_str = time.strftime("%Y-%m-%d %H:%M:%S")
            print(f"\n--- {current_time_str} | State: {self.current_state.value} ---")

            try:
                # --- State transitions based on the defined sequence ---
                if self.current_state == Project2States.FIND_FIRST_BLOCK:
                    self.handle_find_object(Project2States.APPROACH_FIRST_BLOCK, "Block 1")

                elif self.current_state == Project2States.APPROACH_FIRST_BLOCK:
                    self.handle_approach_object(Project2States.GRAB_FIRST_BLOCK, "Block 1")

                elif self.current_state == Project2States.GRAB_FIRST_BLOCK:
                    self.handle_grab(Project2States.LIFT_ARM_AFTER_GRAB1)

                elif self.current_state == Project2States.LIFT_ARM_AFTER_GRAB1:
                    self.handle_move_arm(ARM_LIFT_Y, Project2States.BACKUP_AFTER_GRAB1)

                elif self.current_state == Project2States.BACKUP_AFTER_GRAB1:
                    self.handle_backup(BACKUP_DIST_LONG, Project2States.RELEASE_FIRST_BLOCK_TEMP)

                elif self.current_state == Project2States.RELEASE_FIRST_BLOCK_TEMP:
                    # Temporarily place block 1 down
                    self.handle_release(Project2States.LOWER_ARM_AFTER_RELEASE1)

                elif self.current_state == Project2States.LOWER_ARM_AFTER_RELEASE1:
                     self.handle_move_arm(ARM_LOWER_Y, Project2States.BACKUP_AND_RESET_ARM)

                elif self.current_state == Project2States.BACKUP_AND_RESET_ARM:
                     # Back up further and ensure arm is clear (already lowered)
                     # Gripper is open from previous step.
                     self.handle_backup(BACKUP_DIST_SHORT, Project2States.SURVEY_FOR_BLOCK2)

                elif self.current_state == Project2States.SURVEY_FOR_BLOCK2:
                     self.handle_find_object(Project2States.APPROACH_BLOCK2, "Block 2")

                elif self.current_state == Project2States.APPROACH_BLOCK2:
                     self.handle_approach_object(Project2States.GRAB_BLOCK2, "Block 2")

                elif self.current_state == Project2States.GRAB_BLOCK2:
                     # Grab block 2, then lift it before searching for Target 1
                     self.handle_grab(Project2States.LIFT_ARM_AFTER_GRAB2)

                elif self.current_state == Project2States.LIFT_ARM_AFTER_GRAB2:
                    self.handle_move_arm(ARM_LIFT_Y, Project2States.SURVEY_FOR_TARGET1)

                elif self.current_state == Project2States.SURVEY_FOR_TARGET1:
                     self.handle_find_object(Project2States.APPROACH_TARGET1, "Target 1")

                elif self.current_state == Project2States.APPROACH_TARGET1:
                     self.handle_approach_object(Project2States.RELEASE_BLOCK2_AT_TARGET1, "Target 1")

                elif self.current_state == Project2States.RELEASE_BLOCK2_AT_TARGET1:
                     # Release Block 2 at Target 1
                     self.handle_release(Project2States.LOWER_ARM_AFTER_RELEASE2)

                elif self.current_state == Project2States.LOWER_ARM_AFTER_RELEASE2:
                    # Lower arm after releasing Block 2
                    self.handle_move_arm(ARM_LOWER_Y, Project2States.BACKUP_AFTER_TARGET1)

                elif self.current_state == Project2States.BACKUP_AFTER_TARGET1:
                    # Backup a bit after dropping Block 2
                    self.handle_backup(BACKUP_DIST_VERY_SHORT, Project2States.SURVEY_FOR_BLOCK1_AGAIN)

                elif self.current_state == Project2States.SURVEY_FOR_BLOCK1_AGAIN:
                     # Find the temporarily placed Block 1 again
                     self.handle_find_object(Project2States.APPROACH_BLOCK1_AGAIN, "Block 1")

                elif self.current_state == Project2States.APPROACH_BLOCK1_AGAIN:
                     self.handle_approach_object(Project2States.GRAB_BLOCK1_AGAIN, "Block 1")

                elif self.current_state == Project2States.GRAB_BLOCK1_AGAIN:
                     # Grab Block 1 again
                     self.handle_grab(Project2States.LIFT_ARM_AFTER_GRAB3)

                elif self.current_state == Project2States.LIFT_ARM_AFTER_GRAB3:
                    # Lift Block 1 after grabbing it again
                    self.handle_move_arm(ARM_LIFT_Y, Project2States.SURVEY_FOR_TARGET2)

                elif self.current_state == Project2States.SURVEY_FOR_TARGET2:
                     self.handle_find_object(Project2States.APPROACH_TARGET2, "Target 2")

                elif self.current_state == Project2States.APPROACH_TARGET2:
                     self.handle_approach_object(Project2States.RELEASE_BLOCK1_AT_TARGET2, "Target 2")

                elif self.current_state == Project2States.RELEASE_BLOCK1_AT_TARGET2:
                     # Release Block 1 at Target 2
                     self.handle_release(Project2States.LOWER_ARM_AFTER_RELEASE3)

                elif self.current_state == Project2States.LOWER_ARM_AFTER_RELEASE3:
                     # Lower arm after final release
                     self.handle_move_arm(ARM_LOWER_Y, Project2States.COMPLETED)

                else:
                    # Should not happen if all states are handled
                    print(f"FATAL ERROR: Reached unknown state: {self.current_state}. Stopping.")
                    self.current_state = Project2States.ERROR

                # --- Optional: Update Live View ---
                # Display is handled within find/approach if self.show_video is True
                # If not finding or approaching, show last known frame briefly
                if self.show_video and self.last_vis_frame is not None and \
                   self.current_state not in [Project2States.FIND_FIRST_BLOCK, Project2States.APPROACH_FIRST_BLOCK,
                                             Project2States.SURVEY_FOR_BLOCK2, Project2States.APPROACH_BLOCK2,
                                             Project2States.SURVEY_FOR_TARGET1, Project2States.APPROACH_TARGET1,
                                             Project2States.SURVEY_FOR_BLOCK1_AGAIN, Project2States.APPROACH_BLOCK1_AGAIN,
                                             Project2States.SURVEY_FOR_TARGET2, Project2States.APPROACH_TARGET2]:
                     cv2.imshow('Robot View', self.last_vis_frame)
                     if cv2.waitKey(1) & 0xFF == ord('q'):
                          print("Manual quit requested.")
                          self.current_state = Project2States.ERROR
                          break # Exit while loop immediately

                # Control loop frequency slightly
                loop_duration = time.time() - start_loop_time
                sleep_time = max(0.01, 0.05 - loop_duration) # Aim for roughly 20Hz max, but don't sleep too long
                # time.sleep(sleep_time)


            except KeyboardInterrupt:
                print("\nKeyboard interrupt detected. Stopping.")
                self.current_state = Project2States.ERROR
                break # Exit while loop
            except Exception as e:
                print(f"\n--------------------")
                print(f"FATAL ERROR occurred in state {self.current_state}:")
                print(f"Error Type: {type(e).__name__}")
                print(f"Error Details: {e}")
                print("Traceback:")
                traceback.print_exc()
                print(f"--------------------")
                self.current_state = Project2States.ERROR
                break # Exit while loop

        # End of loop
        if self.current_state == Project2States.COMPLETED:
            print("\n==============================================")
            print("--- Project 2 Sequence Completed Successfully! ---")
            print("==============================================")
            # Optional final action, e.g., celebratory spin
            # try: self.ep_chassis.move(z=360, z_speed=45).wait_for_completed()
            # except: pass
        else:
            print("\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            print("--- Project 2 Sequence Ended with Error or Interruption ---")
            print(f"--- Final State: {self.current_state.value} ---")
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

        self.cleanup()


if __name__ == '__main__':
    # --- Configuration ---
    # Replace with your robot's Serial Number, or leave as None/"" to auto-discover
    # Auto-discovery only works if only one robot is on the network.
    ROBOT_SN = "3JKCH8800100YN"
    SHOW_LIVE_VIDEO = True # Set to False to disable the OpenCV window

    # --- Execution ---
    state_machine = Project2StateMachine(robot_sn=ROBOT_SN)
    state_machine.show_video = SHOW_LIVE_VIDEO
    state_machine.run()