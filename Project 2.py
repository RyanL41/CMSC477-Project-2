import time
import cv2
import numpy as np
from ultralytics import YOLO
from robomaster import robot, camera
from enum import Enum
import traceback
import matplotlib.pyplot as plt
import os

YOLO_MODEL_PATH = "best.pt"
TARGET_BBOX_HEIGHT_APPROACH = 220
TARGET_BBOX_HEIGHT_APPROACH_2 = 191
TARGET_BBOX_HEIGHT_APPROACH_3 = 192

TARGET_Y_POSITIONS = {
    "Block 1": TARGET_BBOX_HEIGHT_APPROACH,
    "Block 2": TARGET_BBOX_HEIGHT_APPROACH_2,
    "Target 1": TARGET_BBOX_HEIGHT_APPROACH,
    "Target 2": TARGET_BBOX_HEIGHT_APPROACH_3,
}
TARGET_X_CENTER = 320


class Project2States(Enum):
    INITIALIZING = "initializing"
    FIND_FIRST_BLOCK = "find_first_block"
    APPROACH_FIRST_BLOCK = "approach_first_block"
    GRAB_FIRST_BLOCK = "grab_first_block"
    LIFT_ARM_AFTER_GRAB1 = "lift_arm_after_grab1"
    BACKUP_AFTER_GRAB1 = "backup_after_grab1"
    RELEASE_FIRST_BLOCK_TEMP = "release_first_block_temp"
    LOWER_ARM_AFTER_RELEASE1 = "lower_arm_after_release1"
    BACKUP_AND_RESET_ARM = "backup_and_reset_arm"
    SURVEY_FOR_BLOCK2 = "survey_for_block2"
    APPROACH_BLOCK2 = "approach_block2"
    GRAB_BLOCK2 = "grab_block2"
    LIFT_ARM_AFTER_GRAB2 = "lift_arm_after_grab2"
    SURVEY_FOR_TARGET1 = "survey_for_target1"
    APPROACH_TARGET1 = "approach_target1"
    RELEASE_BLOCK2_AT_TARGET1 = "release_block2_at_target1"
    LOWER_ARM_AFTER_RELEASE2 = "lower_arm_after_release2"
    BACKUP_AFTER_TARGET1 = "backup_after_target1"
    SURVEY_FOR_BLOCK1_AGAIN = "survey_for_block1_again"
    APPROACH_BLOCK1_AGAIN = "approach_block1_again"
    GRAB_BLOCK1_AGAIN = "grab_block1_again"
    LIFT_ARM_AFTER_GRAB3 = "lift_arm_after_grab3"
    SURVEY_FOR_TARGET2 = "survey_for_target2"
    APPROACH_TARGET2 = "approach_target2"
    RELEASE_BLOCK1_AT_TARGET2 = "release_block1_at_target2"
    LOWER_ARM_AFTER_RELEASE3 = "lower_arm_after_release3"
    COMPLETED = "completed"
    ERROR = "error"


class Project2StateMachine:
    def __init__(self, robot_sn):
        self.robot_sn = robot_sn
        self.ep_robot = None
        self.yolo_model = YOLO(YOLO_MODEL_PATH)

        self.current_state = Project2States.INITIALIZING
        self.target_label = None
        self.last_detection = None
        self.last_vis_frame = None

        self.approach_plot_data = {}

    def initialize_robot(self):
        try:
            self.ep_robot = robot.Robot()
            self.ep_robot.initialize(conn_type="sta", sn=self.robot_sn)

            self.ep_robot.camera.start_video_stream(
                display=False, resolution=camera.STREAM_360P
            )

            self.ep_robot.robotic_arm.move(x=180, y=-70).wait_for_completed()
            time.sleep(0.5)
            self.ep_robot.gripper.open(power=70)
            time.sleep(1)
            self.ep_robot.gripper.pause()

            self.current_state = Project2States.FIND_FIRST_BLOCK
            print("Robot Initialized Successfully")
        except Exception as e:
            print(f"Error during initialization: {e}")
            traceback.print_exc()
            self.current_state = Project2States.ERROR

    def reset_robot(self):
        print("Resetting robot...")
        cv2.destroyAllWindows()
        if self.ep_robot:
            try:
                if self.ep_robot.camera._active:
                    self.ep_robot.camera.stop_video_stream()
            except Exception as e:
                print(f" Minor error stopping video stream: {e}")

            if self.ep_robot.chassis:
                self.ep_robot.chassis.drive_speed(x=0, y=0, z=0)
                time.sleep(0.1)
            if self.ep_robot.gripper:
                self.ep_robot.gripper.pause()
            try:

                print("Resetting arm position...")
                self.ep_robot.robotic_arm.recenter().wait_for_completed(timeout=5)
                time.sleep(1)
                self.ep_robot.robotic_arm.move(y=-70).wait_for_completed(timeout=5)
                time.sleep(1)
            except Exception as e:
                print(f"Could not reset arm position: {e}")

            self.ep_robot.close()
            print("Robot connection closed.")
        else:
            print("No robot object to reset.")

    def get_frame(self):
        if not self.ep_robot or not self.ep_robot.camera._active:
            print("Camera not active, cannot get frame.")
            return None
        try:
            frame = self.ep_robot.camera.read_cv2_image(strategy="newest", timeout=0.5)

            if frame is None:

                time.sleep(0.05)
            return frame
        except Exception as e:
            print(f"Exception getting frame: {e}")
            time.sleep(0.1)
            return None

    def run_yolo_detection(self, frame):
        if frame is None or self.yolo_model is None:

            return [], frame

        try:
            results = self.yolo_model.predict(
                source=frame, show=False, verbose=False, conf=0.70, imgsz=320
            )[0]

            boxes = results.boxes
            class_names = self.yolo_model.names
            vis_frame = results.plot()
            detections_list = []

            for box in boxes:
                xyxy = box.xyxy.cpu().numpy().flatten().astype(int)
                class_id = int(box.cls.cpu().numpy())
                label = class_names[class_id]
                confidence = float(box.conf.cpu().numpy())

                detections_list.append(
                    {"label": label, "confidence": confidence, "box": xyxy}
                )

            self.last_vis_frame = vis_frame
            return detections_list, vis_frame
        except Exception as e:
            print(f"Error during YOLO detection: {e}")
            traceback.print_exc()
            return [], frame

    def get_target_label_detection(self, target_label, detections):
        if not detections:
            return None

        best_detection = None

        if "Block" in target_label:
            max_y2 = -1
            for det in detections:
                if det["label"] == target_label:
                    _, _, _, y2 = det["box"]

                    if y2 > max_y2:
                        max_y2 = y2
                        best_detection = det
        else:
            max_confidence = 0.0
            for det in detections:
                if det["label"] == target_label and det["confidence"] > max_confidence:
                    max_confidence = det["confidence"]
                    best_detection = det

        return best_detection

    def approach_object_simple(self, detection, target_label, current_state_key):

        if not detection:
            print(f"Approach {target_label}: No detection provided.")
            return 0, 0, 0

        x1, y1, x2, y2 = detection["box"]
        box_center_x = (x1 + x2) / 2
        box_height = y2 - y1

        target_y1 = TARGET_Y_POSITIONS.get(target_label, TARGET_BBOX_HEIGHT_APPROACH)

        if current_state_key:
            if current_state_key not in self.approach_plot_data:
                self.approach_plot_data[current_state_key] = {
                    "time_steps": [],
                    "actual_x": [],
                    "target_x": [],
                    "actual_y": [],
                    "target_y": [],
                }
            step = len(self.approach_plot_data[current_state_key]["time_steps"])
            self.approach_plot_data[current_state_key]["time_steps"].append(step)
            self.approach_plot_data[current_state_key]["actual_x"].append(box_center_x)
            self.approach_plot_data[current_state_key]["target_x"].append(
                TARGET_X_CENTER
            )
            self.approach_plot_data[current_state_key]["actual_y"].append(y1)
            self.approach_plot_data[current_state_key]["target_y"].append(target_y1)

        is_close_enough = y1 > target_y1

        if is_close_enough:
            print(
                f"Approach {target_label}: Close enough (y1={y1} > target={target_y1}). Stopping."
            )
            return 0, 0, 0

        error_x = TARGET_X_CENTER - box_center_x
        z_vel = np.clip(error_x * 0.005, -15, 15)

        error_y = target_y1 - y1

        x_vel = np.clip(0.05 + error_y * 0.001, 0.05, 0.2)

        if error_y < 30:
            x_vel = 0.05
            z_vel = np.clip(error_x * 0.004, -10, 10)

        return x_vel, 0, z_vel

    def handle_find_object(self, next_state, search_label):
        print(f"State: Finding {search_label}")
        self.target_label = search_label
        self.ep_robot.chassis.drive_speed(x=0, y=0, z=15)
        found_object = None
        search_start_time = time.time()

        while time.time() - search_start_time < 25:
            frame = self.get_frame()
            if frame is None:
                continue

            detections, _ = self.run_yolo_detection(frame)
            found_object = self.get_target_label_detection(
                self.target_label, detections
            )

            if found_object:
                print(f"Found {search_label}!")
                self.ep_robot.chassis.drive_speed(x=0, y=0, z=0)
                time.sleep(0.2)
                self.last_detection = found_object
                self.current_state = next_state
                return

            if self.last_vis_frame is not None:
                cv2.imshow("Robot View - Finding", self.last_vis_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("Quit command received.")
                    self.current_state = Project2States.ERROR
                    return

        print(f"Could not find {search_label} within time limit.")
        self.ep_robot.chassis.drive_speed(x=0, y=0, z=0)
        self.current_state = Project2States.ERROR

    def handle_approach_object(self, next_state, target_label):

        self.target_label = target_label
        current_state_key = self.current_state.value

        frame = self.get_frame()
        if frame is None:
            time.sleep(0.1)
            return

        detections, _ = self.run_yolo_detection(frame)
        current_detection = self.get_target_label_detection(
            self.target_label, detections
        )

        approach_detection = None
        if current_detection:

            approach_detection = current_detection
            self.last_detection = current_detection
        elif self.last_detection and self.last_detection["label"] == self.target_label:

            print(f"Approach {target_label}: Target lost, stopping momentarily.")
            self.ep_robot.chassis.drive_speed(x=0, y=0, z=0)

            approach_detection = None
            time.sleep(0.2)
        else:

            print(f"Approach {target_label}: Target lost, stopping.")
            self.ep_robot.chassis.drive_speed(x=0, y=0, z=0)
            self.current_state = Project2States.ERROR
            return

        x_vel, y_vel, z_vel = self.approach_object_simple(
            approach_detection, target_label, current_state_key
        )

        if x_vel == 0 and y_vel == 0 and z_vel == 0:
            if approach_detection:
                print(
                    f"Approach complete for {target_label}. Transitioning to {next_state.value}"
                )
                self.ep_robot.chassis.drive_speed(x=0, y=0, z=0)
                time.sleep(0.3)
                self.current_state = next_state
            else:

                print(
                    f"Approach {target_label}: Stopped due to lost target, remaining in approach state."
                )
                self.ep_robot.chassis.drive_speed(x=0, y=0, z=0)
        else:

            self.ep_robot.chassis.drive_speed(x=x_vel, y=y_vel, z=z_vel, timeout=0.5)

        if self.last_vis_frame is not None:

            vis_frame_with_targets = self.last_vis_frame.copy()
            target_y = TARGET_Y_POSITIONS.get(target_label, TARGET_BBOX_HEIGHT_APPROACH)

            cv2.line(
                vis_frame_with_targets, (0, target_y), (640, target_y), (0, 0, 255), 1
            )

            cv2.line(
                vis_frame_with_targets,
                (TARGET_X_CENTER, 0),
                (TARGET_X_CENTER, 360),
                (0, 0, 255),
                1,
            )
            cv2.putText(
                vis_frame_with_targets,
                f"State: {current_state_key}",
                (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 0),
                1,
            )
            cv2.putText(
                vis_frame_with_targets,
                f"Targeting: {target_label}",
                (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 0),
                1,
            )

            cv2.imshow("Robot View - Approaching", vis_frame_with_targets)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("Quit command received.")
                self.current_state = Project2States.ERROR

    def handle_grab(self, next_state):
        print(f"State: Grabbing (Target: {self.target_label})")

        self.ep_robot.gripper.close(power=50)
        time.sleep(1.5)
        self.ep_robot.gripper.pause()
        print("Grab complete.")
        self.current_state = next_state

    def handle_release(self, next_state):
        print(f"State: Releasing (Target: {self.target_label})")
        self.ep_robot.gripper.open(power=50)
        time.sleep(1.5)
        self.ep_robot.gripper.pause()
        print("Release complete.")
        self.current_state = next_state

    def handle_move_arm(self, x_pos, y_pos, next_state):

        print(f"State: Moving arm to x={x_pos}, y={y_pos}")
        try:
            self.ep_robot.robotic_arm.move(x=x_pos, y=y_pos).wait_for_completed(
                timeout=10
            )
            print("Arm move complete.")
            self.current_state = next_state
        except Exception as e:
            print(f"Arm move failed: {e}")
            self.current_state = Project2States.ERROR

    def handle_backup(self, distance_m, next_state):
        print(f"State: Backing up {distance_m}m")
        if distance_m == 0:
            self.current_state = next_state
            return
        try:

            self.ep_robot.chassis.move(
                x=-abs(distance_m), y=0, z=0, xy_speed=0.2
            ).wait_for_completed(timeout=15)
            print("Backup complete.")
            self.current_state = next_state
        except Exception as e:
            print(f"Backup move failed: {e}")
            self.current_state = Project2States.ERROR

    def plot_approach_data(self):

        print("Generating approach plots...")
        plot_dir = "approach_plots"
        os.makedirs(plot_dir, exist_ok=True)

        for state_key, data in self.approach_plot_data.items():
            if not data["time_steps"]:
                print(f"No data to plot for {state_key}")
                continue

            print(f"Plotting for state: {state_key}")
            time_steps = data["time_steps"]
            actual_x = data["actual_x"]
            target_x = data["target_x"]
            actual_y = data["actual_y"]
            target_y = data["target_y"]

            plt.figure(figsize=(10, 5))
            plt.plot(
                time_steps,
                actual_x,
                label="Actual X Center",
                marker="o",
                linestyle="-",
                markersize=4,
            )
            plt.plot(
                time_steps,
                target_x,
                label="Target X Center (320)",
                linestyle="--",
                color="red",
            )
            plt.xlabel("Time Step")
            plt.ylabel("X Pixel Coordinate")
            plt.title(f"Approach Phase: {state_key} - X Position vs Time")
            plt.legend()
            plt.grid(True)
            plot_filename_x = os.path.join(plot_dir, f"{state_key}_x_position.png")
            plt.savefig(plot_filename_x)
            print(f"Saved X plot: {plot_filename_x}")

            plt.figure(figsize=(10, 5))
            plt.plot(
                time_steps,
                actual_y,
                label="Actual Y1 (Top Edge)",
                marker="o",
                linestyle="-",
                markersize=4,
            )
            plt.plot(
                time_steps,
                target_y,
                label=f"Target Y1 ({target_y[0]})",
                linestyle="--",
                color="red",
            )
            plt.xlabel("Time Step")
            plt.ylabel("Y Pixel Coordinate")
            plt.title(f"Approach Phase: {state_key} - Y Position (Top Edge) vs Time")

            plt.gca().invert_yaxis()
            plt.legend()
            plt.grid(True)
            plot_filename_y = os.path.join(plot_dir, f"{state_key}_y_position.png")
            plt.savefig(plot_filename_y)
            print(f"Saved Y plot: {plot_filename_y}")

            plt.close("all")

        print("Finished generating plots.")

    def run(self):

        ARM_GRAB_HEIGHT = -70
        ARM_LIFT_HEIGHT = -30
        ARM_TRAVEL_HEIGHT = 0
        ARM_RELEASE_HEIGHT_TEMP = 0
        ARM_RELEASE_HEIGHT_FINAL = -50

        self.initialize_robot()

        while self.current_state not in [
            Project2States.COMPLETED,
            Project2States.ERROR,
        ]:

            current_time_str = time.strftime("%Y-%m-%d %H:%M:%S")
            print(f"\n--- {current_time_str} | State: {self.current_state.value} ---")

            try:

                if self.current_state == Project2States.FIND_FIRST_BLOCK:
                    self.handle_find_object(
                        Project2States.APPROACH_FIRST_BLOCK, "Block 1"
                    )

                elif self.current_state == Project2States.APPROACH_FIRST_BLOCK:
                    self.handle_approach_object(
                        Project2States.GRAB_FIRST_BLOCK, "Block 1"
                    )

                elif self.current_state == Project2States.GRAB_FIRST_BLOCK:
                    self.handle_grab(Project2States.LIFT_ARM_AFTER_GRAB1)

                elif self.current_state == Project2States.LIFT_ARM_AFTER_GRAB1:

                    self.handle_move_arm(
                        180, ARM_LIFT_HEIGHT, Project2States.BACKUP_AFTER_GRAB1
                    )

                elif self.current_state == Project2States.BACKUP_AFTER_GRAB1:

                    self.handle_move_arm(
                        180, ARM_TRAVEL_HEIGHT, "temp_state_before_backup1"
                    )
                elif self.current_state == "temp_state_before_backup1":
                    self.handle_backup(0.8, Project2States.RELEASE_FIRST_BLOCK_TEMP)

                elif self.current_state == Project2States.RELEASE_FIRST_BLOCK_TEMP:

                    self.handle_move_arm(
                        180, ARM_RELEASE_HEIGHT_TEMP, "temp_state_before_release1"
                    )
                elif self.current_state == "temp_state_before_release1":
                    self.handle_release(Project2States.LOWER_ARM_AFTER_RELEASE1)

                elif self.current_state == Project2States.LOWER_ARM_AFTER_RELEASE1:

                    self.handle_move_arm(
                        180, ARM_GRAB_HEIGHT, Project2States.BACKUP_AND_RESET_ARM
                    )

                elif self.current_state == Project2States.BACKUP_AND_RESET_ARM:

                    self.handle_backup(0.4, Project2States.SURVEY_FOR_BLOCK2)

                elif self.current_state == Project2States.SURVEY_FOR_BLOCK2:
                    self.handle_find_object(Project2States.APPROACH_BLOCK2, "Block 2")

                elif self.current_state == Project2States.APPROACH_BLOCK2:
                    self.handle_approach_object(Project2States.GRAB_BLOCK2, "Block 2")

                elif self.current_state == Project2States.GRAB_BLOCK2:
                    self.handle_grab(Project2States.LIFT_ARM_AFTER_GRAB2)

                elif self.current_state == Project2States.LIFT_ARM_AFTER_GRAB2:

                    self.handle_move_arm(
                        180, ARM_LIFT_HEIGHT, "temp_state_before_survey1"
                    )
                elif self.current_state == "temp_state_before_survey1":

                    self.handle_move_arm(
                        180, ARM_TRAVEL_HEIGHT, Project2States.SURVEY_FOR_TARGET1
                    )

                elif self.current_state == Project2States.SURVEY_FOR_TARGET1:
                    self.handle_find_object(Project2States.APPROACH_TARGET1, "Target 1")

                elif self.current_state == Project2States.APPROACH_TARGET1:
                    self.handle_approach_object(
                        Project2States.RELEASE_BLOCK2_AT_TARGET1, "Target 1"
                    )

                elif self.current_state == Project2States.RELEASE_BLOCK2_AT_TARGET1:

                    self.handle_move_arm(
                        180, ARM_RELEASE_HEIGHT_FINAL, "temp_state_before_release2"
                    )
                elif self.current_state == "temp_state_before_release2":
                    self.handle_release(Project2States.LOWER_ARM_AFTER_RELEASE2)

                elif self.current_state == Project2States.LOWER_ARM_AFTER_RELEASE2:

                    self.handle_move_arm(
                        180, ARM_LIFT_HEIGHT, Project2States.BACKUP_AFTER_TARGET1
                    )

                elif self.current_state == Project2States.BACKUP_AFTER_TARGET1:
                    self.handle_backup(0.3, Project2States.SURVEY_FOR_BLOCK1_AGAIN)

                elif self.current_state == Project2States.SURVEY_FOR_BLOCK1_AGAIN:

                    self.handle_move_arm(
                        180, ARM_GRAB_HEIGHT, "temp_state_before_find1_again"
                    )
                elif self.current_state == "temp_state_before_find1_again":
                    self.handle_find_object(
                        Project2States.APPROACH_BLOCK1_AGAIN, "Block 1"
                    )

                elif self.current_state == Project2States.APPROACH_BLOCK1_AGAIN:
                    self.handle_approach_object(
                        Project2States.GRAB_BLOCK1_AGAIN, "Block 1"
                    )

                elif self.current_state == Project2States.GRAB_BLOCK1_AGAIN:
                    self.handle_grab(Project2States.LIFT_ARM_AFTER_GRAB3)

                elif self.current_state == Project2States.LIFT_ARM_AFTER_GRAB3:

                    self.handle_move_arm(
                        180, ARM_LIFT_HEIGHT, "temp_state_before_survey2"
                    )
                elif self.current_state == "temp_state_before_survey2":

                    self.handle_move_arm(
                        180, ARM_TRAVEL_HEIGHT, Project2States.SURVEY_FOR_TARGET2
                    )

                elif self.current_state == Project2States.SURVEY_FOR_TARGET2:
                    self.handle_find_object(Project2States.APPROACH_TARGET2, "Target 2")

                elif self.current_state == Project2States.APPROACH_TARGET2:
                    self.handle_approach_object(
                        Project2States.RELEASE_BLOCK1_AT_TARGET2, "Target 2"
                    )

                elif self.current_state == Project2States.RELEASE_BLOCK1_AT_TARGET2:

                    self.handle_move_arm(
                        180, ARM_RELEASE_HEIGHT_FINAL, "temp_state_before_release3"
                    )
                elif self.current_state == "temp_state_before_release3":
                    self.handle_release(Project2States.LOWER_ARM_AFTER_RELEASE3)

                elif self.current_state == Project2States.LOWER_ARM_AFTER_RELEASE3:

                    self.handle_move_arm(
                        180, ARM_LIFT_HEIGHT, "temp_state_before_final_backup"
                    )
                elif self.current_state == "temp_state_before_final_backup":

                    self.handle_backup(0.3, Project2States.COMPLETED)

                if (
                    "APPROACH" not in self.current_state.value
                    and "FIND" not in self.current_state.value
                ):
                    time.sleep(0.05)

            except Exception as e:
                print(
                    f"!!!--- Exception occurred in state {self.current_state.value} ---!!!"
                )
                traceback.print_exc()
                self.current_state = Project2States.ERROR

                try:
                    if self.ep_robot and self.ep_robot.chassis:
                        self.ep_robot.chassis.drive_speed(x=0, y=0, z=0, timeout=1)
                except:
                    pass
                break

        print(
            f"\n=== State Machine Finished with State: {self.current_state.value} ==="
        )

        self.plot_approach_data()

        self.reset_robot()


if __name__ == "__main__":

    robot_serial_number = "3JKCH8800100YN"

    state_machine = Project2StateMachine(robot_sn=robot_serial_number)
    state_machine.run()
    print("Script finished.")
