"""
Vision module for object detection using YOLO.
"""
import cv2
import numpy as np
from ultralytics import YOLO
from .config import YOLO_MODEL_PATH


LEGO_BIG_LABEL = "lego_big"
LEGO_SMALL_LABEL = "lego_small"
LEGO_MED_LABEL = "lego_med"


class ObjectDetector:
    def __init__(self, model_path=YOLO_MODEL_PATH):
        """
        Initialize the YOLO object detector.
        
        Args:
            model_path: Path to the YOLO model weights
        """
        self.model = YOLO(model_path)
        self.last_frame = None
        
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

        results = self.model.predict(
            source=frame, show=False, verbose=False, conf=confidence_threshold
        )[0]

        boxes = results.boxes
        class_names = self.model.names
        vis_frame = frame.copy()
        detections_list = []

        for box in boxes:
            xyxy = box.xyxy.cpu().numpy().flatten().astype(int)
            class_id = int(box.cls.cpu().numpy())
            label = class_names[class_id]
            confidence = float(box.conf.cpu().numpy())

            detections_list.append({
                "label": label, 
                "confidence": confidence, 
                "box": xyxy
            })

            # Draw bounding box and label
            self._draw_detection(vis_frame, xyxy, label, confidence)

        self.last_frame = vis_frame
        return detections_list, vis_frame
    
    def _draw_detection(self, frame, box, label, confidence):
        """Draw detection on the frame."""
        cv2.rectangle(
            frame,
            (box[0], box[1]),
            (box[2], box[3]),
            color=(0, 255, 0),
            thickness=2,
        )
        label_text = f"{label} ({confidence:.2f})"
        cv2.putText(
            frame,
            label_text,
            (box[0], box[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
        )
    
    def get_best_detection(self, target_label, detections):
        """
        Get the highest confidence detection of the specified label.
        
        Args:
            target_label: Target object label
            detections: List of detections
            
        Returns:
            Best detection or None if not found
        """
        if not detections or not target_label:
            return None

        best_detection = None
        max_confidence = 0.0

        for detection in detections:
            if detection["label"] == target_label and detection["confidence"] > max_confidence:
                x1, y1, x2, y2 = detection["box"]
                if (x1 > 10 and x1 < 630 and x2 > 10 and x2 < 630):
                    max_confidence = detection["confidence"]
                    best_detection = detection

        return best_detection
