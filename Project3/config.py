"""
Configuration parameters for the Project3 robot control system.
"""
from enum import Enum

# Robot states
class RobotState(Enum):
    INITIALIZING = "initializing"
    LOOKING_FOR_BLOCK_IN_CLOSET = "looking_for_block_in_closet" 
    REMOVE_FROM_CLOSET = "remove_from_closet"
    LOOKING_FOR_BLOCK = "looking_for_block"
    APPROACH_BLOCK = "approach_block"
    GRAB_BLOCK = "grab_block"
    MOVE_ARM = "move_arm"
    DROP_OFF = "drop_off"
    BACKUP = "backup"
    DELIVER_BLOCK = "deliver_block"
    BULLY_MODE = "bully_mode"
    WALL_MODE = "wall_mode"
    LOOKING_FOR_OBSTACLES = "looking_for_obstacles"
    ERROR = "error"

# YOLO model path
YOLO_MODEL_PATH = "best.pt"

# Target bounding box dimensions
TARGET_BBOX_SMALL_HEIGHT = 160
TARGET_BBOX_MEDIUM_HEIGHT = 125
TARGET_BBOX_LARGE_HEIGHT = 192

# Lego block labels
LEGO_BIG_LABEL = "lego_big"
LEGO_SMALL_LABEL = "lego_small"
LEGO_MED_LABEL = "lego_med"
BLOCK_LABELS = [LEGO_BIG_LABEL, LEGO_SMALL_LABEL, LEGO_MED_LABEL]

# Map configuration
STARTING_POSITION_NUMBER = 2  # either 2 or 5 (see InitialMap.csv)
SELF_CLOSET_NUMBER = 4       # either 3 or 4 (see InitialMap.csv)
TARGET_CLOSET_NUMBER = 3     # either 3 or 4 (see InitialMap.csv)
TARGET_SEARCH_NUMBER = 5

# Robot physical parameters
CUBE_SIZE_METERS = 0.26      # 1 cube unit = 0.26 meters
CAMERA_MATRIX = [[314, 0, 320], [0, 314, 180], [0, 0, 1]]
APRILTAG_SIZE_METERS = 0.153  # Size of the AprilTag in meters
SCALE_FACTOR = 0.266        # scaling factor from image units to meters
