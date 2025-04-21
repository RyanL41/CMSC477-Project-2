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
#NEED TO DETERMINE THESE VALUES
TARGET_BBOX_SMALL_HEIGHT_APPROACH = 160
TARGET_BBOX_MEDIUM_HEIGHT_APPROACH = 125
TARGET_BBOX_LARGE_HEIGHT_APPROACH = 192


class Project3States(Enum):
    INITIALIZING = "initializing"
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



class Project3StateMachine:
    def __init__(self, robot_sn):
        self.robot_sn = robot_sn
        self.ep_robot = robot.Robot()
        self.yolo_model = YOLO(YOLO_MODEL_PATH)

        self.current_state = Project3States.INITIALIZING
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
            for state in Project3States
        }

    # FIND_FIRST_BLOCK = "find_first_block"
    # APPROACH_FIRST_BLOCK = "approach_first_block"
    # GRAB_FIRST_BLOCK = "grab_first_block"
    # LIFT_ARM_AFTER_GRAB1 = "lift_arm_after_grab1"
    # BACKUP_AFTER_GRAB1 = "backup_after_grab1"
    # RELEASE_FIRST_BLOCK_TEMP = "release_first_block_temp"
    # LOWER_ARM_AFTER_RELEASE1 = "lower_arm_after_release1"
    # BACKUP_AND_RESET_ARM = "backup_and_reset_arm"
    # SURVEY_FOR_BLOCK2 = "survey_for_block2"
    # APPROACH_BLOCK2 = "approach_block2"
    # GRAB_BLOCK2 = "grab_block2"
    # LIFT_ARM_AFTER_GRAB2 = "lift_arm_after_grab2"
    # SURVEY_FOR_TARGET1 = "survey_for_target1"
    # APPROACH_TARGET1 = "approach_target1"
    # RELEASE_BLOCK2_AT_TARGET1 = "release_block2_at_target1"
    # LOWER_ARM_AFTER_RELEASE2 = "lower_arm_after_release2"
    # BACKUP_AFTER_TARGET1 = "backup_after_target1"
    # SURVEY_FOR_BLOCK1_AGAIN = "survey_for_block1_again"
    # APPROACH_BLOCK1_AGAIN = "approach_block1_again"
    # GRAB_BLOCK1_AGAIN = "grab_block1_again"
    # LIFT_ARM_AFTER_GRAB3 = "lift_arm_after_grab3"
    # SURVEY_FOR_TARGET2 = "survey_for_target2"
    # APPROACH_TARGET2 = "approach_target2"
    # RELEASE_BLOCK1_AT_TARGET2 = "release_block1_at_target2"
    # LOWER_ARM_AFTER_RELEASE3 = "lower_arm_after_release3"
    # COMPLETED = "completed"
    # ERROR = "error"



    def initialize_robot(self):
        self.ep_robot.initialize(conn_type="sta", sn=self.robot_sn)

        self.ep_robot.camera.start_video_stream(
            display=False, resolution=camera.STREAM_360P
        )
