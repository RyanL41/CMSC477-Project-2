from ultralytics import YOLO
import cv2
import time
from robomaster import robot
from robomaster import camera


print('model')
model = YOLO("best.pt")


# Initialize the robot and camera
ep_robot = robot.Robot()
ep_robot.initialize(conn_type="sta", sn="3JKCH8800100YN")
ep_camera = ep_robot.camera
ep_camera.start_video_stream(display=False, resolution=camera.STREAM_360P)


while True:
   # Read frame from the robot camera
   frame = ep_camera.read_cv2_image(strategy="newest", timeout=0.5)
   if frame is not None:
       start = time.time()
      
       # Perform detection
       result = model.predict(source=frame, show=False)[0]


       # DIY visualization (faster than using show=True)
       boxes = result.boxes
       class_names = model.names  # Class names corresponding to label IDs
      
       for box in boxes:
           xyxy = box.xyxy.cpu().numpy().flatten()  # Bounding box coordinates
           class_id = int(box.cls.cpu().numpy())  # Class ID of the detection
        #    label = class_names[class_id]  # Class name for the given class ID
        #    confidence = box.conf.cpu().numpy()  # Confidence score of the detection


           # Draw bounding box
           cv2.rectangle(frame,
                         (int(xyxy[0]), int(xyxy[1])),
                         (int(xyxy[2]), int(xyxy[3])),
                         color=(0, 0, 255), thickness=2)


           # Add label and confidence score
        #    label_text = f"{label} ({confidence:.2f})"
           label_text = str(class_id)
           cv2.putText(frame, label_text, (int(xyxy[0]), int(xyxy[1]) - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)


       # Display the frame
       cv2.imshow('frame', frame)


       # Break if 'q' is pressed
       key = cv2.waitKey(1)
       if key == ord('q'):
           break


       # Print the detection result (optional for debugging)
       print(result)


       # Measure frame rate
       end = time.time()
       print(1.0 / (end - start))