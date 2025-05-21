"""
Common utility functions for the P3-Revamp robot control system.
"""
import time
import cv2
import numpy as np
import threading
from P3_Revamp.config import (
    CAMERA_CENTER_X, CAMERA_CENTER_Y, DEFAULT_SPEED, 
    DISTANCE_THRESHOLD, CENTER_THRESHOLD
)

# Global lock for thread safety
global_lock = threading.Lock()

def create_thread_lock():
    """
    Create a new thread lock.
    
    Returns:
        threading.Lock: A new thread lock
    """
    return threading.Lock()

def preprocess_frame(frame):
    """
    Preprocess a camera frame for vision processing.
    
    Args:
        frame: Input camera frame
        
    Returns:
        Preprocessed frame or None if frame is invalid
    """
    if frame is None:
        return None
    
    # Check if frame is valid
    if frame.size == 0 or frame.shape[0] == 0 or frame.shape[1] == 0:
        return None
    
    # Resize frame if needed (not necessary if camera already provides correct size)
    # height, width = frame.shape[:2]
    # if width != 640 or height != 360:
    #     frame = cv2.resize(frame, (640, 360))
    
    return frame

def calculate_distance(pos1, pos2):
    """
    Calculate Euclidean distance between two positions.
    
    Args:
        pos1: First position as (x, y) tuple
        pos2: Second position as (x, y) tuple
        
    Returns:
        Distance between positions
    """
    return np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)

def calculate_angle_to_target(current_pos, target_pos):
    """
    Calculate angle (in degrees) from current position to target position.
    
    Args:
        current_pos: Current position as (x, y, theta) tuple where theta is in radians
        target_pos: Target position as (x, y) tuple
        
    Returns:
        Angle to target in degrees
    """
    current_x, current_y, current_heading = current_pos
    target_x, target_y = target_pos
    
    # Calculate angle to target
    dx = target_x - current_x
    dy = target_y - current_y
    
    # Calculate angle in radians
    target_angle = np.arctan2(dy, dx)
    
    # Convert to degrees
    target_angle_deg = np.rad2deg(target_angle)
    current_heading_deg = np.rad2deg(current_heading)
    
    # Calculate angle difference
    angle_diff = (target_angle_deg - current_heading_deg) % 360
    if angle_diff > 180:
        angle_diff -= 360
    
    return angle_diff

def calculate_movement_vector(current_pos, target_pos, max_distance=0.5):
    """
    Calculate movement vector from current position to target position.
    
    Args:
        current_pos: Current position as (x, y) tuple
        target_pos: Target position as (x, y) tuple
        max_distance: Maximum distance to move in one step
        
    Returns:
        Movement vector as (dx, dy) tuple
    """
    current_x, current_y = current_pos
    target_x, target_y = target_pos
    
    # Calculate distance and direction
    dx = target_x - current_x
    dy = target_y - current_y
    distance = np.sqrt(dx**2 + dy**2)
    
    # Cap movement at max_distance
    if distance > max_distance:
        ratio = max_distance / distance
        dx *= ratio
        dy *= ratio
    
    return (dx, dy)

def is_target_reached(current_pos, target_pos, threshold=DISTANCE_THRESHOLD):
    """
    Check if the target position has been reached.
    
    Args:
        current_pos: Current position as (x, y) tuple or (x, y, theta) tuple
        target_pos: Target position as (x, y) tuple
        threshold: Distance threshold to consider target reached
        
    Returns:
        Boolean indicating whether target is reached
    """
    # Extract x, y coordinates from current_pos
    current_x, current_y = current_pos[0], current_pos[1]
    target_x, target_y = target_pos
    
    # Calculate distance to target
    distance = np.sqrt((target_x - current_x)**2 + (target_y - current_y)**2)
    
    return distance < threshold

def is_centered(detection, threshold=CENTER_THRESHOLD):
    """
    Check if the detected object is centered in the frame.
    
    Args:
        detection: Detection dictionary with 'box' or 'center' key
        threshold: Pixel threshold to consider object centered
        
    Returns:
        Boolean indicating whether object is centered
    """
    if detection is None:
        return False
    
    # Get center of detection
    if 'center' in detection:
        center_x, center_y = detection['center']
    else:
        x1, y1, x2, y2 = detection['box']
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
    
    # Calculate distance from camera center
    distance_from_center = abs(center_x - CAMERA_CENTER_X)
    
    return distance_from_center < threshold

def is_detection_recent(detection, max_age=0.5):
    """
    Check if the detection is recent.
    
    Args:
        detection: Detection dictionary with 'timestamp' key
        max_age: Maximum age in seconds to consider detection recent
        
    Returns:
        Boolean indicating whether detection is recent
    """
    if detection is None or 'timestamp' not in detection:
        return False
    
    return time.time() - detection['timestamp'] < max_age

def create_shared_data_structure():
    """
    Create a shared data structure for communication between threads.
    
    Returns:
        Dictionary with shared data
    """
    return {
        'apriltags': {},  # Tag ID -> {relative_position, world_position, timestamp}
        'objects': {},    # Label -> detection
        'target_found': False,
        'last_update': 0,
        'position': (0, 0, 0)
    }

def safe_get_detection(shared_data, label, lock=None):
    """
    Safely get detection data from shared data structure.
    
    Args:
        shared_data: Shared data dictionary
        label: Object label to get
        lock: Optional thread lock to use
        
    Returns:
        Detection dictionary or None if not found
    """
    if lock:
        with lock:
            if label in shared_data['objects']:
                return shared_data['objects'][label]
    else:
        if label in shared_data['objects']:
            return shared_data['objects'][label]
    
    return None

def safe_get_apriltag(shared_data, tag_id, lock=None):
    """
    Safely get AprilTag data from shared data structure.
    
    Args:
        shared_data: Shared data dictionary
        tag_id: AprilTag ID to get
        lock: Optional thread lock to use
        
    Returns:
        AprilTag data dictionary or None if not found
    """
    if lock:
        with lock:
            if tag_id in shared_data['apriltags']:
                return shared_data['apriltags'][tag_id]
    else:
        if tag_id in shared_data['apriltags']:
            return shared_data['apriltags'][tag_id]
    
    return None

def safe_set_detection(shared_data, label, detection, lock=None):
    """
    Safely set detection data in shared data structure.
    
    Args:
        shared_data: Shared data dictionary
        label: Object label to set
        detection: Detection data to set
        lock: Optional thread lock to use
    """
    if lock:
        with lock:
            shared_data['objects'][label] = detection
            shared_data['last_update'] = time.time()
    else:
        shared_data['objects'][label] = detection
        shared_data['last_update'] = time.time()

def safe_set_apriltag(shared_data, tag_id, tag_data, lock=None):
    """
    Safely set AprilTag data in shared data structure.
    
    Args:
        shared_data: Shared data dictionary
        tag_id: AprilTag ID to set
        tag_data: AprilTag data to set
        lock: Optional thread lock to use
    """
    if lock:
        with lock:
            shared_data['apriltags'][tag_id] = tag_data
            shared_data['last_update'] = time.time()
    else:
        shared_data['apriltags'][tag_id] = tag_data
        shared_data['last_update'] = time.time()

def get_box_dimensions(detection):
    """
    Get dimensions of a detection bounding box.
    
    Args:
        detection: Detection dictionary with 'box' key
        
    Returns:
        (width, height) tuple or None if detection is None
    """
    if detection is None:
        return None
    
    x1, y1, x2, y2 = detection['box']
    width = x2 - x1
    height = y2 - y1
    
    return (width, height)

def get_box_center(detection):
    """
    Get center coordinates of a detection bounding box.
    
    Args:
        detection: Detection dictionary with 'box' key
        
    Returns:
        (center_x, center_y) tuple or None if detection is None
    """
    if detection is None:
        return None
    
    if 'center' in detection:
        return detection['center']
    
    x1, y1, x2, y2 = detection['box']
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    
    return (center_x, center_y)

def get_rotation_direction(box_center_x, camera_center_x=CAMERA_CENTER_X):
    """
    Determine rotation direction to center an object.
    
    Args:
        box_center_x: X-coordinate of box center
        camera_center_x: X-coordinate of camera center
        
    Returns:
        Integer: Positive for clockwise, negative for counter-clockwise
    """
    error = camera_center_x - box_center_x
    return np.sign(error)  # 1 for positive, -1 for negative

def log_debug(message, debug=False):
    """
    Log a debug message if debug mode is enabled.
    
    Args:
        message: Message to log
        debug: Whether debug mode is enabled
    """
    if debug:
        current_time = time.strftime("%Y-%m-%d %H:%M:%S")
        print(f"[DEBUG] {current_time}: {message}")

def log_info(message):
    """
    Log an info message.
    
    Args:
        message: Message to log
    """
    current_time = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[INFO] {current_time}: {message}")