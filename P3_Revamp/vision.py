"""
Vision module for object detection using YOLO.
"""
import cv2
import numpy as np
import time
from ultralytics import YOLO
from P3_Revamp.config import (
    YOLO_MODEL_PATH, BLOCK_LABELS, LEGO_BIG_LABEL, 
    LEGO_SMALL_LABEL, LEGO_MED_LABEL, CENTER_LINE_LABEL, 
    CLOSET_LABEL, ROBOT_LABEL, CAMERA_CENTER_X, CAMERA_CENTER_Y
)

class ObjectDetector:
    def __init__(self, model_path=YOLO_MODEL_PATH, debug=False):
        """
        Initialize the YOLO object detector.
        
        Args:
            model_path: Path to the YOLO model weights
            debug: Enable debug mode with additional logging and visualization
        """
        self.model = YOLO(model_path)
        self.last_frame = None
        self.debug = debug
        self.last_detection_time = 0
        
    def get_detections(self, frame, confidence_threshold=0.70):
        """
        Run object detection on the frame.
        
        Args:
            frame: Input image
            confidence_threshold: Minimum confidence for detections
            
        Returns:
            List of detections, visualized frame
        """
        if frame is None or self.model is None:
            return [], None

        try:
            results = self.model.predict(
                source=frame, show=False, verbose=False, conf=confidence_threshold
            )[0]
            
            boxes = results.boxes
            class_names = self.model.names
            vis_frame = frame.copy() if self.debug else None
            detections_list = []

            for box in boxes:
                xyxy = box.xyxy.cpu().numpy().flatten().astype(int)
                class_id = int(box.cls.cpu().numpy())
                label = class_names[class_id]
                confidence = float(box.conf.cpu().numpy())

                # Calculate center point of detection
                center_x = (xyxy[0] + xyxy[2]) / 2
                center_y = (xyxy[1] + xyxy[3]) / 2
                
                # Calculate distance from center of frame
                center_dist = np.sqrt(
                    (center_x - CAMERA_CENTER_X)**2 + 
                    (center_y - CAMERA_CENTER_Y)**2
                )

                detections_list.append({
                    "label": label, 
                    "confidence": confidence, 
                    "box": xyxy,
                    "center": (center_x, center_y),
                    "center_distance": center_dist,
                    "timestamp": time.time()
                })

                # Draw bounding box and label if in debug mode
                if self.debug and vis_frame is not None:
                    self._draw_detection(vis_frame, xyxy, label, confidence)

            self.last_frame = vis_frame
            self.last_detection_time = time.time()
            return detections_list, vis_frame
            
        except Exception as e:
            print(f"Error during object detection: {e}")
            return [], None
    
    def _draw_detection(self, frame, box, label, confidence):
        """
        Draw detection on the frame.
        
        Args:
            frame: Image to draw on
            box: Bounding box coordinates [x1, y1, x2, y2]
            label: Object label
            confidence: Detection confidence
        """
        # Get color based on label
        if label in BLOCK_LABELS:
            color = (0, 255, 0)  # Green for lego blocks
        elif label == CENTER_LINE_LABEL:
            color = (255, 0, 0)  # Blue for center line
        elif label == CLOSET_LABEL:
            color = (0, 0, 255)  # Red for closet
        elif label == ROBOT_LABEL:
            color = (255, 255, 0)  # Cyan for robot
        else:
            color = (255, 0, 255)  # Magenta for other objects
            
        # Draw bounding box
        cv2.rectangle(
            frame,
            (box[0], box[1]),
            (box[2], box[3]),
            color=color,
            thickness=2,
        )
        
        # Draw label with confidence
        label_text = f"{label} ({confidence:.2f})"
        cv2.putText(
            frame,
            label_text,
            (box[0], box[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
        )
        
        # Draw center point
        center_x = (box[0] + box[2]) // 2
        center_y = (box[1] + box[3]) // 2
        cv2.circle(frame, (center_x, center_y), 5, color, -1)
    
    def get_best_detection(self, target_label, detections, by_confidence=True):
        """
        Get the best detection of the specified label.
        
        Args:
            target_label: Target object label or list of labels
            detections: List of detections
            by_confidence: If True, get highest confidence detection.
                           If False, get detection closest to center of frame.
            
        Returns:
            Best detection or None if not found
        """
        if not detections:
            return None
            
        # Convert single label to list if needed
        if isinstance(target_label, str):
            target_labels = [target_label]
        else:
            target_labels = target_label
            
        # Filter detections by target labels
        matching_detections = [
            d for d in detections if d["label"] in target_labels
        ]
        
        if not matching_detections:
            return None
            
        if by_confidence:
            # Return detection with highest confidence
            return max(matching_detections, key=lambda d: d["confidence"])
        else:
            # Return detection closest to center of frame
            return min(matching_detections, key=lambda d: d["center_distance"])
    
    def get_detection_dimensions(self, detection):
        """
        Get the dimensions of a detection's bounding box.
        
        Args:
            detection: Detection dictionary
            
        Returns:
            (width, height) tuple or None if detection is None
        """
        if detection is None:
            return None
            
        x1, y1, x2, y2 = detection["box"]
        width = x2 - x1
        height = y2 - y1
        
        return (width, height)
        
    def get_detection_center(self, detection):
        """
        Get the center coordinates of a detection.
        
        Args:
            detection: Detection dictionary
            
        Returns:
            (center_x, center_y) tuple or None if detection is None
        """
        if detection is None:
            return None
            
        if "center" in detection:
            return detection["center"]
            
        x1, y1, x2, y2 = detection["box"]
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        
        return (center_x, center_y)