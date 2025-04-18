import os
import time
import cv2
import numpy as np
from ultralytics import YOLO
from robomaster import robot, camera
from enum import Enum
import traceback
from matplotlib import pyplot as plt


YOLO_MODEL_PATH = "best.pt"
TARGET_BBOX_HEIGHT_APPROACH = 160
TARGET_BBOX_HEIGHT_APPROACH_2 = 125
TARGET_BBOX_HEIGHT_APPROACH_3 = 192


TARGET_Y_POSITIONS = {
    "Block 1": TARGET_BBOX_HEIGHT_APPROACH,
    "Block 2": TARGET_BBOX_HEIGHT_APPROACH_2,
    "Target 1": TARGET_BBOX_HEIGHT_APPROACH,
    "Target 2": TARGET_BBOX_HEIGHT_APPROACH_3,
}

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
        self.ep_robot = robot.Robot()
        self.yolo_model = YOLO(YOLO_MODEL_PATH)

        self.current_state = Project2States.INITIALIZING
        self.target_label = None
        self.last_detection = None
        self.last_vis_frame = None
        self.approach_plot_data = {
            state.value: {
                "time_steps": [],
                "actual_x": [],
                "target_x": [],
                "actual_y": [],
                "target_y": [],
            }
            for state in Project2States
        }

    def initialize_robot(self):

        self.ep_robot.initialize(conn_type="sta", sn=self.robot_sn)

        self.ep_robot.camera.start_video_stream(
            display=False, resolution=camera.STREAM_360P
        )

        self.ep_robot.robotic_arm.move(x=0, y=-100).wait_for_completed()

        self.ep_robot.gripper.open(power=70)

        time.sleep(1)
        self.ep_robot.gripper.pause()
        
        self.current_state = Project2States.FIND_FIRST_BLOCK

    def reset_robot(self):
        cv2.destroyAllWindows()
        if self.ep_robot.camera:
            self.ep_robot.camera.stop_video_stream()
        if self.ep_robot:
            if self.ep_robot.chassis:
                self.ep_robot.chassis.drive_speed(x=0, y=0, z=0)
            if self.ep_robot.gripper:
                self.ep_robot.gripper.pause()
            self.ep_robot.close()

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
                320
            )
            self.approach_plot_data[current_state_key]["actual_y"].append(y1)
            self.approach_plot_data[current_state_key]["target_y"].append(target_y1)

        print(y1, target_y1)

        is_close_enough = y1 > target_y1
        is_kinda_close = y1 > target_y1 - 50

        if is_close_enough:
            print(
                f"Approach {target_label}: Close enough (y1={y1} > target={target_y1}). Stopping."
            )
            return 0, 0, 0

        error_x = 320 - box_center_x
        z_vel = np.clip(-error_x * 0.1, -25, 25)

        if is_kinda_close:
            return 0.04, 0, z_vel

        return 0.1, 0, z_vel


    def handle_find_object(self, next_state, search_label):

        self.target_label = search_label
        self.ep_robot.chassis.drive_speed(x=0, y=0, z=20)

        while True:

            frame = self.get_frame()
            detections, _ = self.run_yolo_detection(frame)
            found_object = self.get_target_label_detection(
                self.target_label, detections
            )

            if found_object:
                self.ep_robot.chassis.drive_speed(x=0, y=0, z=0)
                self.last_detection = found_object
                self.current_state = next_state
                return

            if self.last_vis_frame is not None:
                cv2.imshow("Robot View", self.last_vis_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    self.current_state = Project2States.ERROR
                    return

    def handle_approach_object(self, next_state, target_label):

        self.target_label = target_label
        current_state_key = self.current_state.value

        frame = self.get_frame()
        detections, _ = self.run_yolo_detection(frame)
        current_detection = self.get_target_label_detection(
            self.target_label, detections
        )

        approach_detection = None
        if current_detection:
            approach_detection = current_detection
            self.last_detection = current_detection
        elif self.last_detection and self.last_detection["label"] == self.target_label:
            approach_detection = self.last_detection
        else:
            self.ep_robot.chassis.drive_speed(x=0, y=0, z=0)
            self.current_state = Project2States.ERROR
            return

        x_vel, y_vel, z_vel = self.approach_object_simple(
            approach_detection, target_label, current_state_key
        )

        if x_vel == 0 and y_vel == 0 and z_vel == 0:
            self.ep_robot.chassis.drive_speed(x=0, y=0, z=0)
            self.current_state = next_state
        else:
            self.ep_robot.chassis.drive_speed(x=x_vel, y=y_vel, z=z_vel)

        if self.last_vis_frame is not None:
            cv2.imshow("Robot View", self.last_vis_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                self.current_state = Project2States.ERROR

    def handle_grab(self, next_state):
        self.ep_robot.gripper.close(power=70)
        time.sleep(1.5)
        self.ep_robot.gripper.pause()
        self.current_state = next_state

    def handle_release(self, next_state):
        self.ep_robot.gripper.open(power=70)
        time.sleep(1.5)
        self.ep_robot.gripper.pause()
        self.current_state = next_state

    def handle_move_arm(self, y_distance, next_state):
        self.ep_robot.robotic_arm.move(y=y_distance).wait_for_completed(timeout=10)
        self.current_state = next_state

    def handle_backup(self, distance_m, next_state):
        self.ep_robot.chassis.move(
            x=-distance_m, y=0, z=0, xy_speed=0.1
        ).wait_for_completed(timeout=15)
        self.current_state = next_state

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
                    self.handle_move_arm(20, Project2States.BACKUP_AFTER_GRAB1)

                elif self.current_state == Project2States.BACKUP_AFTER_GRAB1:
                    self.handle_backup(1, Project2States.RELEASE_FIRST_BLOCK_TEMP)

                elif self.current_state == Project2States.RELEASE_FIRST_BLOCK_TEMP:
                    self.handle_release(Project2States.LOWER_ARM_AFTER_RELEASE1)

                elif self.current_state == Project2States.LOWER_ARM_AFTER_RELEASE1:
                    self.handle_move_arm(-100, Project2States.BACKUP_AND_RESET_ARM)

                elif self.current_state == Project2States.BACKUP_AND_RESET_ARM:
                    self.handle_backup(0.5, Project2States.SURVEY_FOR_BLOCK2)

                elif self.current_state == Project2States.SURVEY_FOR_BLOCK2:
                    self.handle_find_object(Project2States.APPROACH_BLOCK2, "Block 2")

                elif self.current_state == Project2States.APPROACH_BLOCK2:
                    self.handle_approach_object(Project2States.GRAB_BLOCK2, "Block 2")

                elif self.current_state == Project2States.GRAB_BLOCK2:
                    self.handle_grab(Project2States.LIFT_ARM_AFTER_GRAB2)

                elif self.current_state == Project2States.LIFT_ARM_AFTER_GRAB2:
                    self.handle_move_arm(20, Project2States.SURVEY_FOR_TARGET1)

                elif self.current_state == Project2States.SURVEY_FOR_TARGET1:
                    self.handle_find_object(Project2States.APPROACH_TARGET1, "Target 1")

                elif self.current_state == Project2States.APPROACH_TARGET1:
                    self.handle_approach_object(
                        Project2States.RELEASE_BLOCK2_AT_TARGET1, "Target 1"
                    )

                elif self.current_state == Project2States.RELEASE_BLOCK2_AT_TARGET1:
                    self.handle_release(Project2States.LOWER_ARM_AFTER_RELEASE2)

                elif self.current_state == Project2States.LOWER_ARM_AFTER_RELEASE2:
                    self.handle_move_arm(-20, Project2States.BACKUP_AFTER_TARGET1)

                elif self.current_state == Project2States.BACKUP_AFTER_TARGET1:
                    self.handle_backup(0.25, Project2States.SURVEY_FOR_BLOCK1_AGAIN)

                elif self.current_state == Project2States.SURVEY_FOR_BLOCK1_AGAIN:
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
                    self.handle_move_arm(20, Project2States.SURVEY_FOR_TARGET2)

                elif self.current_state == Project2States.SURVEY_FOR_TARGET2:
                    self.handle_find_object(Project2States.APPROACH_TARGET2, "Target 2")

                elif self.current_state == Project2States.APPROACH_TARGET2:
                    self.handle_approach_object(
                        Project2States.RELEASE_BLOCK1_AT_TARGET2, "Target 2"
                    )

                elif self.current_state == Project2States.RELEASE_BLOCK1_AT_TARGET2:
                    self.handle_release(Project2States.LOWER_ARM_AFTER_RELEASE3)

                elif self.current_state == Project2States.LOWER_ARM_AFTER_RELEASE3:
                    self.handle_move_arm(-20, Project2States.COMPLETED)

                else:
                    print(self.current_state)
                    self.current_state = Project2States.ERROR

                if self.last_vis_frame is not None and self.current_state not in [
                    Project2States.FIND_FIRST_BLOCK,
                    Project2States.APPROACH_FIRST_BLOCK,
                    Project2States.SURVEY_FOR_BLOCK2,
                    Project2States.APPROACH_BLOCK2,
                    Project2States.SURVEY_FOR_TARGET1,
                    Project2States.APPROACH_TARGET1,
                    Project2States.SURVEY_FOR_BLOCK1_AGAIN,
                    Project2States.APPROACH_BLOCK1_AGAIN,
                    Project2States.SURVEY_FOR_TARGET2,
                    Project2States.APPROACH_TARGET2,
                ]:
                    cv2.imshow("Robot View", self.last_vis_frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        self.current_state = Project2States.ERROR
                        break

            except Exception as e:
                traceback.print_exc()
                self.current_state = Project2States.ERROR
                break

        print(
            f"\n=== State Machine Finished with State: {self.current_state.value} ==="
        )

        self.plot_approach_data()

        self.reset_robot()


if __name__ == "__main__":

    Project2StateMachine(robot_sn="3JKCH8800100YN").run()
