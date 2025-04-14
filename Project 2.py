import time
import cv2
import numpy as np
from ultralytics import YOLO
from robomaster import robot, camera
from enum import Enum
import traceback


YOLO_MODEL_PATH = "best.pt"
TARGET_BBOX_WIDTH_APPROACH = 82
TARGET_BBOX_WIDTH_APPROACH_2 = 80
TARGET_BBOX_WIDTH_APPROACH_3 = 260
TARGET_BBOX_WIDTH_APPROACH_4 = 87
TARGET_BBOX_HEIGHT_APPROACH = 220
TARGET_BBOX_HEIGHT_APPROACH_2 = 191
TARGET_BBOX_HEIGHT_APPROACH_3 = 192


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

    def approach_object_simple(self, detection, target_label):
        global TARGET_BBOX_WIDTH_APPROACH, TARGET_BBOX_WIDTH_APPROACH_2, TARGET_BBOX_WIDTH_APPROACH_3, TARGET_BBOX_WIDTH_APPROACH_4

        if not detection:
            return 0, 0, 0

        x1, y1, x2, y2 = detection["box"]

        box_center_x = (x1 + x2) / 2
        box_width = x2 - x1

        is_close_enough = (
            y1 > TARGET_BBOX_HEIGHT_APPROACH_2
            if target_label == "Block 2"
            else (y1 > TARGET_BBOX_HEIGHT_APPROACH_3)
            if target_label == "Target 2"
            else y1 > TARGET_BBOX_HEIGHT_APPROACH
        )

        if is_close_enough:
            return 0, 0, 0

        thresh = 5

        if target_label == "Block 1":
            if box_width > (TARGET_BBOX_WIDTH_APPROACH - thresh):
                TARGET_BBOX_WIDTH_APPROACH -= 0.05

        elif target_label == "Block 2":
            if box_width > (TARGET_BBOX_WIDTH_APPROACH_2 - thresh):
                TARGET_BBOX_WIDTH_APPROACH_2 -= 0.1

        elif (
            self.current_state == Project2States.GRAB_BLOCK1_AGAIN
            or self.current_state == Project2States.APPROACH_TARGET2
        ):
            if box_width > (TARGET_BBOX_WIDTH_APPROACH_3 - thresh):
                TARGET_BBOX_WIDTH_APPROACH_3 -= 0.05
        else:
            if box_width > (TARGET_BBOX_WIDTH_APPROACH_4 - thresh):
                TARGET_BBOX_WIDTH_APPROACH_4 -= 0.05

        error_x = 320 - box_center_x
        z_vel = np.clip(-error_x * 0.1, -25, 25)
        x_vel = 0.1
        if y1 > 170:
            x_vel = 0.05
        return x_vel, 0, z_vel

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
            approach_detection, target_label
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

        self.reset_robot()


if __name__ == "__main__":

    Project2StateMachine(robot_sn="3JKCH8800100YN").run()
