"""
Configuration parameters for the P3-Revamp robot control system.
"""
from enum import Enum

# Robot states
class RobotState(Enum):
    LOCALIZATION = "localization"
    PATH_PLANNING_1_LEGO = "path_planning_1_lego"
    LEGO_PICKUP = "lego_pickup"
    PATH_PLANNING_2_CENTER_LINE_PRE = "path_planning_2_center_line_pre"
    INTERMEDIARY_CENTERING = "intermediary_centering"
    PATH_PLANNING_3_CENTER_LINE_POST = "path_planning_3_center_line_post"
    LEGO_DROPOFF = "lego_dropoff"
    PATH_PLANNING_4_CENTER_LINE_POST_DROPOFF = "path_planning_4_center_line_post_dropoff"
    PATH_PLANNING_5_CENTER_LINE_RETURN = "path_planning_5_center_line_return"
    ERROR = "error"

# YOLO model path
YOLO_MODEL_PATH = "best.pt"

# Initial robot position
INITIAL_POSITION = (2, -2, 0)  # x, y, theta

# Target positions
LEGO_SEARCH_POSITION = (10, -4)
CENTER_LINE_WAYPOINT = (6, -3)

# Target bounding box dimensions
TARGET_BBOX_SMALL_HEIGHT = 160
TARGET_BBOX_MEDIUM_HEIGHT = 125
TARGET_BBOX_LARGE_HEIGHT = 192

# Camera parameters
CAMERA_MATRIX = [[314, 0, 320], [0, 314, 180], [0, 0, 1]]
CAMERA_CENTER_X = 320
CAMERA_CENTER_Y = 180

# Lego block labels
LEGO_BIG_LABEL = "lego_big"
LEGO_SMALL_LABEL = "lego_small"
LEGO_MED_LABEL = "lego_med"
BLOCK_LABELS = [LEGO_BIG_LABEL, LEGO_SMALL_LABEL, LEGO_MED_LABEL]

# Other object labels
CENTER_LINE_LABEL = "center_line"
CLOSET_LABEL = "closet"
ROBOT_LABEL = "robot"

# AprilTag parameters
APRILTAG_SIZE_METERS = 0.153  # Size of the AprilTag in meters
APRILTAG_FAMILY = "tag36h11"
APRILTAG_THREADS = 2
APRILTAG_PROXIMITY_THRESHOLD = 1.0  # Box units for obstacle avoidance

# Robot physical parameters
CUBE_SIZE_METERS = 0.26      # 1 cube unit = 0.26 meters
SCALE_FACTOR = 0.266        # scaling factor from image units to meters

# Movement parameters
DEFAULT_SPEED = 0.3
SLOW_SPEED = 0.1
ROTATION_SPEED = 15  # degrees per rotation
MAX_SINGLE_MOVE_DISTANCE = 0.5  # meters

# Approach parameters
LEGO_PICKUP_Y_THRESHOLD = 230  # Y-position threshold for lego pickup
CENTER_THRESHOLD = 20  # Pixels from center to consider centered
DISTANCE_THRESHOLD = 0.5  # Meters from target to consider reached

# Obstacle avoidance parameters
OBSTACLE_AVOIDANCE_DISTANCE = 0.3  # Lateral movement distance for obstacle avoidance

# Gripper parameters
GRIPPER_POWER = 70
GRIPPER_DELAY = 1.5  # seconds

# Arm parameters
ARM_DOWN_POSITION = -100
ARM_UP_POSITION = 70

# Thread and lock parameters
VISION_THREAD_SLEEP = 0.01  # seconds
MOVEMENT_THREAD_SLEEP = 0.1  # seconds
DETECTION_FRESHNESS_THRESHOLD = 0.5  # seconds

# Search parameters
ROTATION_SEARCH_INTERVAL = 10.0  # seconds between rotation searches
SEARCH_ROTATION_STEP = 15  # degrees per step in rotation search