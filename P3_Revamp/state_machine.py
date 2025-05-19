"""
State machine implementation for the P3-Revamp robot control system.
Coordinates all components to execute the robot's behavior.
"""
import time
import threading
import traceback
import cv2
import numpy as np

from P3_Revamp.config import (
    RobotState, CAMERA_MATRIX, APRILTAG_SIZE_METERS,
    BLOCK_LABELS, CENTER_LINE_LABEL, CLOSET_LABEL,
    LEGO_SEARCH_POSITION, CENTER_LINE_WAYPOINT,
    VISION_THREAD_SLEEP, MOVEMENT_THREAD_SLEEP
)
from P3_Revamp.robot_controller import RobotController
from P3_Revamp.vision import ObjectDetector
from P3_Revamp.apriltag_detector import AprilTagDetector
from P3_Revamp.localization import Localizer
from P3_Revamp.path_planning import PathPlanner
from P3_Revamp.lego_pickup import EnhancedLegoPickupController
from P3_Revamp.lego_dropoff import LegoDropoffController
from P3_Revamp.center_line import CenterLineController
from P3_Revamp.utilities import (
    create_thread_lock, create_shared_data_structure,
    log_debug, log_info
)

class RobotStateMachine:
    def __init__(self, robot_sn, debug=False):
        """
        Initialize the state machine.
        
        Args:
            robot_sn: Serial number of the robot
            debug: Enable debug mode with additional logging
        """
        self.current_state = RobotState.LOCALIZATION
        self.debug = debug
        
        # Initialize robot controller
        self.robot = RobotController(robot_sn, debug=debug)
        
        # Initialize vision systems
        self.object_detector = ObjectDetector(debug=debug)
        self.apriltag_detector = AprilTagDetector(
            np.array(CAMERA_MATRIX), 
            marker_size_m=APRILTAG_SIZE_METERS
        )
        
        # Initialize component controllers
        self.localizer = Localizer(self.robot, self.apriltag_detector, debug=debug)
        self.path_planner = PathPlanner(self.robot, self.object_detector, self.apriltag_detector, debug=debug)
        self.lego_pickup = EnhancedLegoPickupController(self.robot, self.object_detector, self.apriltag_detector, debug=debug)
        self.lego_dropoff = LegoDropoffController(self.robot, self.object_detector, debug=debug)
        self.center_line = CenterLineController(self.robot, self.object_detector, debug=debug)
        
        # Thread control
        self.vision_thread_active = False
        self.movement_enabled = True
        self.target_found = False
        
        # Shared data for thread communication
        self.detection_data = create_shared_data_structure()
        
        # Target tracking
        self.target_labels = []
        self.target_coordinate = None
        self.saved_target_position_intermediary = None
        self.saved_target_position_post = None
        
        # Performance tracking data
        self.performance_data = {
            state.value: {
                "start_time": None,
                "end_time": None,
                "success": False
            }
            for state in RobotState
        }
        
        log_info("State machine initialized")
    
    def vision_loop(self):
        """
        Main vision processing loop - NEVER issues movement commands directly.
        Only updates shared data structures.
        """
        log_info("Vision thread started")
        
        while self.vision_thread_active:
            try:
                # Get current frame
                frame = self.robot.get_frame()
                if frame is None:
                    time.sleep(VISION_THREAD_SLEEP)
                    continue
                
                # Process AprilTags for localization
                self.localizer.detect_and_track_apriltags(
                    frame=frame, 
                    shared_data=self.detection_data,
                )
                
                # Process objects with YOLO
                detections, vis_frame = self.object_detector.get_detections(frame)
                
                # Update shared data with object detections
                if detections:
                    self.detection_data['objects'] = {}
                    for detection in detections:
                        label = detection['label']
                        self.detection_data['objects'][label] = detection
                    
                    self.detection_data['last_update'] = time.time()
            
                # If in debug mode and there's a visualized frame, display it
                if self.debug and vis_frame is not None:
                    cv2.imshow("Vision", vis_frame)
                    cv2.waitKey(1)
                
                # Small sleep to prevent CPU hogging
                time.sleep(VISION_THREAD_SLEEP)
                
            except Exception as e:
                log_info(f"Error in vision loop: {e}")
                if self.debug:
                    traceback.print_exc()
                time.sleep(VISION_THREAD_SLEEP * 10)  # Longer sleep after error
        
        log_info("Vision thread stopped")
        
        # Clean up OpenCV windows if in debug mode
        if self.debug:
            cv2.destroyAllWindows()
    
    def run(self):
        """Main state machine loop."""
        log_info("Starting state machine")
        
        # Initialize robot hardware
        self.robot.initialize()
        
        # Start vision thread
        self.vision_thread_active = True
        vision_thread = threading.Thread(target=self.vision_loop)
        vision_thread.daemon = True
        vision_thread.start()
        
        # Main control loop
        while self.current_state != RobotState.ERROR:
            try:
                # Print current state
                current_time = time.strftime("%Y-%m-%d %H:%M:%S")
                log_info(f"State: {self.current_state.value}")
                
                # Record state start time
                if self.performance_data[self.current_state.value]["start_time"] is None:
                    self.performance_data[self.current_state.value]["start_time"] = time.time()
                
                # Handle states and transitions
                if self.current_state == RobotState.LOCALIZATION:
                    # Execute localization behavior
                    localization_success = self.handle_localization()
                    
                    # STATE TRANSITION: Localization → Path Planning 1 (Lego)
                    if localization_success == "success":
                        log_info("Localization successful! Transitioning to Path Planning 1 (Lego)")
                        
                        # Record state end time and success
                        self.performance_data[self.current_state.value]["end_time"] = time.time()
                        self.performance_data[self.current_state.value]["success"] = True
                        
                        # Transition to next state
                        self.current_state = RobotState.PATH_PLANNING_1_LEGO
                        
                        # Set up parameters for the next state
                        self.target_labels = BLOCK_LABELS
                        self.target_coordinate = LEGO_SEARCH_POSITION
                        self.target_found = False
                        self.path_planner.set_target(self.target_labels, self.target_coordinate)
                
                elif self.current_state == RobotState.PATH_PLANNING_1_LEGO:
                    # Execute path planning behavior
                    path_planning_result = self.handle_path_planning_1_lego()
                    
                    # STATE TRANSITION: Path Planning 1 → Lego Pickup
                    if path_planning_result == "target_found":
                        log_info("Found target object! Transitioning to Lego Pickup")
                        
                        # Record state end time and success
                        self.performance_data[self.current_state.value]["end_time"] = time.time()
                        self.performance_data[self.current_state.value]["success"] = True
                        
                        # Prepare for pickup state
                        self.robot.drive_speed(x=0, y=0, z=0)
                        
                        # Transition to next state
                        self.current_state = RobotState.LEGO_PICKUP
                
                elif self.current_state == RobotState.LEGO_PICKUP:
                    # Execute lego pickup behavior
                    pickup_result = self.handle_lego_pickup()
                    
                    # STATE TRANSITION: Lego Pickup → Path Planning 2 (Center Line Pre)
                    if pickup_result == "success":
                        log_info("Lego pickup successful! Transitioning to Path Planning 2 (Center Line Pre)")
                        
                        # Record state end time and success
                        self.performance_data[self.current_state.value]["end_time"] = time.time()
                        self.performance_data[self.current_state.value]["success"] = True
                        
                        # Set up parameters for the next state
                        self.target_labels = [CENTER_LINE_LABEL]
                        self.target_coordinate = CENTER_LINE_WAYPOINT
                        self.target_found = False
                        self.path_planner.set_target(self.target_labels, self.target_coordinate)
                        
                        # Transition to next state
                        self.current_state = RobotState.PATH_PLANNING_2_CENTER_LINE_PRE
                
                elif self.current_state == RobotState.PATH_PLANNING_2_CENTER_LINE_PRE:
                    # Execute path planning behavior for finding center line
                    path_planning_result = self.handle_path_planning_2_center_line_pre()
                    
                    # STATE TRANSITION: Path Planning 2 → Intermediary Centering
                    if path_planning_result in ["target_found", "position_reached"]:
                        log_info("Found center line or reached target position! Transitioning to Intermediary Centering")
                        
                        # Record state end time and success
                        self.performance_data[self.current_state.value]["end_time"] = time.time()
                        self.performance_data[self.current_state.value]["success"] = True
                        
                        # Prepare for centering state
                        self.robot.drive_speed(x=0, y=0, z=0)
                        
                        # Transition to next state
                        self.current_state = RobotState.INTERMEDIARY_CENTERING
                
                elif self.current_state == RobotState.INTERMEDIARY_CENTERING:
                    # Execute centering behavior
                    centering_result, saved_position = self.handle_intermediary_centering()
                    
                    # STATE TRANSITION: Intermediary Centering → Path Planning 3 (Center Line Post)
                    if centering_result == "success":
                        log_info("Centering successful! Transitioning to Path Planning 3 (Center Line Post)")
                        
                        # Record state end time and success
                        self.performance_data[self.current_state.value]["end_time"] = time.time()
                        self.performance_data[self.current_state.value]["success"] = True
                        
                        # Save the current position for later
                        self.saved_target_position_intermediary = saved_position
                        
                        # Set up parameters for the next state
                        self.target_labels = [CLOSET_LABEL]
                        current_x, current_y, _ = self.robot.get_position()
                        self.target_coordinate = (current_x, current_y + 15)
                        self.saved_target_position_post = (current_x, current_y + 15)
                        self.target_found = False
                        self.path_planner.set_target(self.target_labels, self.target_coordinate)
                        
                        # Transition to next state
                        self.current_state = RobotState.PATH_PLANNING_3_CENTER_LINE_POST
                
                elif self.current_state == RobotState.PATH_PLANNING_3_CENTER_LINE_POST:
                    # Execute path planning behavior for finding closet
                    path_planning_result = self.handle_path_planning_3_center_line_post()
                    
                    # STATE TRANSITION: Path Planning 3 → Lego Dropoff
                    if path_planning_result == "target_found":
                        log_info("Found closet! Transitioning to Lego Dropoff")
                        
                        # Record state end time and success
                        self.performance_data[self.current_state.value]["end_time"] = time.time()
                        self.performance_data[self.current_state.value]["success"] = True
                        
                        # Prepare for dropoff state
                        self.robot.drive_speed(x=0, y=0, z=0)
                        
                        # Transition to next state
                        self.current_state = RobotState.LEGO_DROPOFF
                
                elif self.current_state == RobotState.LEGO_DROPOFF:
                    # Execute lego dropoff behavior
                    dropoff_result = self.handle_lego_dropoff()
                    
                    # STATE TRANSITION: Lego Dropoff → Path Planning 4 (Center Line Post Dropoff)
                    if dropoff_result == "success":
                        log_info("Lego dropoff successful! Transitioning to Path Planning 4 (Center Line Post Dropoff)")
                        
                        # Record state end time and success
                        self.performance_data[self.current_state.value]["end_time"] = time.time()
                        self.performance_data[self.current_state.value]["success"] = True
                        
                        # Set up parameters for the next state
                        self.target_labels = [CENTER_LINE_LABEL]
                        self.target_coordinate = self.saved_target_position_post
                        self.target_found = False
                        self.path_planner.set_target(self.target_labels, self.target_coordinate)
                        
                        # Transition to next state
                        self.current_state = RobotState.PATH_PLANNING_4_CENTER_LINE_POST_DROPOFF
                
                elif self.current_state == RobotState.PATH_PLANNING_4_CENTER_LINE_POST_DROPOFF:
                    # Execute path planning behavior for finding center line post-dropoff
                    path_planning_result = self.handle_path_planning_4_center_line_post_dropoff()
                    
                    # STATE TRANSITION: Path Planning 4 → Path Planning 5 (Center Line Return)
                    if path_planning_result in ["target_found", "position_reached"]:
                        log_info("Found center line or reached target position! Transitioning to Path Planning 5 (Center Line Return)")
                        
                        # Record state end time and success
                        self.performance_data[self.current_state.value]["end_time"] = time.time()
                        self.performance_data[self.current_state.value]["success"] = True
                        
                        # Set up parameters for the next state
                        self.target_labels = [CENTER_LINE_LABEL]
                        self.target_coordinate = self.saved_target_position_intermediary
                        self.target_found = False
                        self.path_planner.set_target(self.target_labels, self.target_coordinate)
                        
                        # Transition to next state
                        self.current_state = RobotState.PATH_PLANNING_5_CENTER_LINE_RETURN
                
                elif self.current_state == RobotState.PATH_PLANNING_5_CENTER_LINE_RETURN:
                    # Execute path planning behavior for returning to starting position
                    path_planning_result = self.handle_path_planning_5_center_line_return()
                    
                    # STATE TRANSITION: Path Planning 5 → Path Planning 1 (Lego) (cycle back)
                    if path_planning_result in ["target_found", "position_reached"]:
                        log_info("Reached starting position! Transitioning back to Path Planning 1 (Lego)")
                        
                        # Record state end time and success
                        self.performance_data[self.current_state.value]["end_time"] = time.time()
                        self.performance_data[self.current_state.value]["success"] = True
                        
                        # Set up parameters for the first state again
                        self.target_labels = BLOCK_LABELS
                        self.target_coordinate = LEGO_SEARCH_POSITION
                        self.target_found = False
                        self.path_planner.set_target(self.target_labels, self.target_coordinate)
                        
                        # Transition to next state (cycle back)
                        self.current_state = RobotState.PATH_PLANNING_1_LEGO
                
                # Add a small sleep to avoid CPU hogging
                time.sleep(MOVEMENT_THREAD_SLEEP)
                
            except KeyboardInterrupt:
                log_info("Keyboard interrupt detected, stopping...")
                break
                
            except Exception as e:
                log_info(f"Error in state machine: {e}")
                if self.debug:
                    traceback.print_exc()
                self.current_state = RobotState.ERROR
                break
        
        # Stop vision thread
        self.vision_thread_active = False
        vision_thread.join(timeout=1.0)
        
        # Print performance data
        if self.debug:
            self.print_performance_data()
        
        log_info(f"State machine finished with state: {self.current_state.value}")
        self.robot.cleanup()
    
    def handle_localization(self):
        """
        Handle the localization state.
        The robot will find an april tag, store its position, and calculate
        its own position relative to the april tag.
        
        Returns:
            "success" if localization was successful, None otherwise
        """
        log_info("Executing localization")
        
        # Run the localization loop
        success = self.localizer.localization_loop()
        
        # If localization was successful, return success
        if success:
            return "success"
        
        return None
    
    def handle_path_planning_1_lego(self):
        """
        Handle the path planning 1 state for finding lego blocks.
        The robot will move toward the target coordinate while looking for lego blocks.
        
        Returns:
            "target_found" if a lego block was found, None otherwise
        """
        log_info(f"Executing path planning for lego blocks: targets={self.target_labels}, position={self.target_coordinate}")
        
        # Run the path planning loop
        found_object, detection = self.path_planner.path_planning_loop()
        
        # If a target object was found, return success
        if found_object and detection is not None:
            log_info(f"Found target object: {detection['label']}")
            return "target_found"
        
        return None
    
    def handle_lego_pickup(self):
        """
        Handle the lego pickup state.
        The robot will center, approach, and pick up a lego block.
        
        Returns:
            "success" if pickup was successful, None otherwise
        """
        log_info("Executing lego pickup")
        
        # Disable movement during pickup operations
        self.robot.set_movement_enabled(False)
        
        # Run the lego pickup loop
        result = self.lego_pickup.lego_pickup_loop()
        
        # Re-enable movement after pickup
        self.robot.set_movement_enabled(True)
        
        # Handle the enhanced result format ("success:label")
        if result and result.startswith("success:"):
            block_label = result.split(":", 1)[1]
            log_info(f"Successfully picked up block: {block_label}")
            return "success"
        
        return result
    
    def handle_path_planning_2_center_line_pre(self):
        """
        Handle the path planning 2 state for finding the center line.
        The robot will move toward the target coordinate while looking for the center line.
        
        Returns:
            "target_found" if center line was found, "position_reached" if target position reached, None otherwise
        """
        log_info(f"Executing path planning for center line pre-waypoint: targets={self.target_labels}, position={self.target_coordinate}")
        
        # Run the path planning loop
        found_object, detection = self.path_planner.path_planning_loop()
        
        # If a target object was found, return target_found
        if found_object and detection is not None:
            log_info(f"Found center line")
            return "target_found"
        
        # Check if we've reached the target position
        current_x, current_y, _ = self.robot.get_position()
        distance_to_target = np.sqrt(
            (current_x - self.target_coordinate[0])**2 + 
            (current_y - self.target_coordinate[1])**2
        )
        
        if distance_to_target < 0.5:
            log_info(f"Reached target position: {self.target_coordinate}")
            return "position_reached"
        
        return None
    
    def handle_intermediary_centering(self):
        """
        Handle the intermediary centering state.
        The robot will rotate to 270 degrees and center on the center line.
        
        Returns:
            Tuple of ("success", target_position) if successful, (None, None) otherwise
        """
        log_info("Executing intermediary centering on center line")
        
        # Disable movement during centering operations
        self.robot.set_movement_enabled(False)
        
        # Run the center line pre-waypoint loop
        result, target_position = self.center_line.center_line_pre_waypoint_loop()
        
        # Re-enable movement after centering
        self.robot.set_movement_enabled(True)
        
        if result == "success":
            log_info(f"Centered on center line at position: {target_position}")
            return "success", target_position
        
        return None, None
    
    def handle_path_planning_3_center_line_post(self):
        """
        Handle the path planning 3 state for finding the closet.
        The robot will move toward the target coordinate while looking for the closet.
        
        Returns:
            "target_found" if closet was found, None otherwise
        """
        log_info(f"Executing path planning for closet: targets={self.target_labels}, position={self.target_coordinate}")
        
        # Run the path planning loop
        found_object, detection = self.path_planner.path_planning_loop()
        
        # If a target object was found, return target_found
        if found_object and detection is not None:
            log_info(f"Found closet")
            return "target_found"
        
        return None
    
    def handle_lego_dropoff(self):
        """
        Handle the lego dropoff state.
        The robot will center, approach, and drop off the lego block at the closet.
        
        Returns:
            "success" if dropoff was successful, None otherwise
        """
        log_info("Executing lego dropoff")
        
        # Disable movement during dropoff operations
        self.robot.set_movement_enabled(False)
        
        # Run the lego dropoff loop
        result = self.lego_dropoff.lego_dropoff_loop()
        
        # Re-enable movement after dropoff
        self.robot.set_movement_enabled(True)
        
        return result
    
    def handle_path_planning_4_center_line_post_dropoff(self):
        """
        Handle the path planning 4 state for finding the center line post-dropoff.
        The robot will move toward the target coordinate while looking for the center line.
        
        Returns:
            "target_found" if center line was found, "position_reached" if target position reached, None otherwise
        """
        log_info(f"Executing path planning for center line post-dropoff: targets={self.target_labels}, position={self.target_coordinate}")
        
        # Run the path planning loop
        found_object, detection = self.path_planner.path_planning_loop()
        
        # If a target object was found, return target_found
        if found_object and detection is not None:
            log_info(f"Found center line")
            return "target_found"
        
        # Check if we've reached the target position
        current_x, current_y, _ = self.robot.get_position()
        distance_to_target = np.sqrt(
            (current_x - self.target_coordinate[0])**2 + 
            (current_y - self.target_coordinate[1])**2
        )
        
        if distance_to_target < 0.5:
            log_info(f"Reached target position: {self.target_coordinate}")
            return "position_reached"
        
        return None
    
    def handle_path_planning_5_center_line_return(self):
        """
        Handle the path planning 5 state for returning to the starting position.
        The robot will move toward the target coordinate while looking for the center line.
        
        Returns:
            "target_found" if center line was found, "position_reached" if target position reached, None otherwise
        """
        log_info(f"Executing path planning for return to start: targets={self.target_labels}, position={self.target_coordinate}")
        
        # Run the path planning loop
        found_object, detection = self.path_planner.path_planning_loop()
        
        # If a target object was found, return target_found
        if found_object and detection is not None:
            log_info(f"Found center line")
            return "target_found"
        
        # Check if we've reached the target position
        current_x, current_y, _ = self.robot.get_position()
        distance_to_target = np.sqrt(
            (current_x - self.target_coordinate[0])**2 + 
            (current_y - self.target_coordinate[1])**2
        )
        
        if distance_to_target < 0.5:
            log_info(f"Reached target position: {self.target_coordinate}")
            return "position_reached"
        
        return None
    
    def print_performance_data(self):
        """Print performance data for all states."""
        log_info("\n===== Performance Data =====")
        
        for state, data in self.performance_data.items():
            if data["start_time"] is not None:
                if data["end_time"] is not None:
                    duration = data["end_time"] - data["start_time"]
                    success = "Success" if data["success"] else "Failed"
                    log_info(f"{state}: {duration:.2f}s - {success}")
                else:
                    log_info(f"{state}: Incomplete")
        
        log_info("=============================")