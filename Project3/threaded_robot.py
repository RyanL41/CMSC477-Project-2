# Run with python3 Project3/threaded_robot.py

#!/usr/bin/env python3
import threading
import time
import cv2
from robomaster import robot
from Project3.vision import ObjectDetector
from Project3.state_machine import RobotStateMachine

class SharedDetections:
    def __init__(self):
        self.lock = threading.Lock()
        self.detections = None
        self.frame = None
        self.terminate = False
    def update(self, detections, frame):
        with self.lock:
            self.detections = detections
            self.frame = frame
    def get(self):
        with self.lock:
            return self.detections, self.frame
    def should_terminate(self):
        with self.lock:
            return self.terminate
    def signal_terminate(self):
        with self.lock:
            self.terminate = True

def vision_thread(shared, ep_robot):
    print("[Vision] Thread started")
    detector = ObjectDetector()
    ep_camera = ep_robot.camera
    ep_camera.start_video_stream(display=False)
    try:
        while not shared.should_terminate():
            frame = ep_camera.read_cv2_image(strategy="newest", timeout=1.0)
            if frame is None:
                time.sleep(0.01)
                continue
            detections, _ = detector.get_detections(frame)
            shared.update(detections, frame)
            time.sleep(0.01)
    except Exception as e:
        print(f"[Vision] Exception: {e}")
    finally:
        ep_camera.stop_video_stream()
        print("[Vision] Thread exiting")

def state_machine_thread(shared, robot_sn, map_file):
    print("[StateMachine] Thread started")
    sm = RobotStateMachine(robot_sn, map_file)
    sm.initialize()  # or whatever your init routine is
    try:
        while not shared.should_terminate():
            detections, frame = shared.get()
            # Optionally, you can pass detections/frame to your state machine here
            # For now, just run the normal state machine logic
            sm.handle_looking_for_block_in_closet()  # or your main loop function
            time.sleep(0.05)
    except Exception as e:
        print(f"[StateMachine] Exception: {e}")
        shared.signal_terminate()
    finally:
        print("[StateMachine] Thread exiting")

def main():
    ep_robot = robot.Robot()
    ep_robot.initialize(conn_type="sta")
    shared = SharedDetections()
    robot_sn = "001"  # replace with your robot serial or config
    map_file = "InitialMap.csv"  # replace with your map file path
    vt = threading.Thread(target=vision_thread, args=(shared, ep_robot), daemon=True)
    st = threading.Thread(target=state_machine_thread, args=(shared, robot_sn, map_file), daemon=True)
    vt.start()
    st.start()
    try:
        while vt.is_alive() and st.is_alive():
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("[Main] KeyboardInterrupt, shutting down...")
        shared.signal_terminate()
    vt.join(timeout=2.0)
    st.join(timeout=2.0)
    ep_robot.close()
    print("[Main] Robot shutdown complete.")

if __name__ == "__main__":
    main()
