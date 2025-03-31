import pupil_apriltags
import cv2
import numpy as np
import time
import traceback
from queue import Empty
from robomaster import robot
from robomaster import camera
from scipy.spatial.transform import Rotation as R
from matplotlib import pyplot as plt
import os

class PWTest:
    
    def __init__(self, K, marker_size_m=0.16):
        self.camera_params = [K[0, 0], K[1, 1], K[0, 2], K[1, 2]]
        self.marker_size_m = marker_size_m
        self.detector = pupil_apriltags.Detector(families="tag36h11", nthreads=2)
        # Tag world positions (x, y) in meters.
        self.tag_id_map = {
            30: (2, 2.5), 31: (2, 3.5), 32: (4, 2.5), 33: (4, 3.5), 34: (5.5, 3),
            35: (1.5, 5), 36: (1.5, 7), 37: (4.5, 6), 38: (6, 5.5), 39: (6, 6.5),
            40: (8, 5.5), 41: (8, 6.5), 42: (2, 8.5), 43: (2, 9), 44: (4, 8.5),
            45: (4, 9.5), 46: (5.5, 9),
            0: (0, 3), 3: (4, 3)
        }
        # Tag world poses: (x, y, yaw_degrees) where yaw is counter-clockwise.
        self.tag_id_map_rot = {
            30: (2, 2.5, 0),
            31: (2, 3.5, 180),
            32: (4, 2.5, 0),
            33: (4, 3.5, 180),
            34: (5.5, 3, 90),
            35: (1.5, 5, 90),
            36: (1.5, 7, 90),
            37: (4.5, 6, 270),
            38: (6, 5.5, 0),
            39: (6, 6.5, 180),
            40: (8, 5.5, 0),
            41: (8, 6.5, 180),
            42: (2, 8.5, 0),
            43: (2, 9.5, 180),
            44: (4, 8.5, 0),
            45: (4, 9.5, 180),
            46: (5.5, 9, 90),
        }

    def get_pose_apriltag_in_camera_frame(self, detection):
        R_ca = detection.pose_R
        t_ca = detection.pose_t
        return t_ca.flatten(), R_ca

    def get_apriltag_id(self, detection):
        return detection.tag_id
    


    def pw(self, detection):
        """
        Computes the robot's position (and overall yaw) in the world frame, adjusting for orientation
        to make it "parallel" to the detected AprilTag.
        """
        tag_id = self.get_apriltag_id(detection)
        if tag_id not in self.tag_id_map or tag_id not in self.tag_id_map_rot:
            return None

        # Get tag pose in camera frame
        t_ca, R_ca = self.get_pose_apriltag_in_camera_frame(detection)
        scale = 0.266  # scaling factor from image units to meters

        # Compute initial offset
        offset = np.array([-t_ca[0] / scale, t_ca[2] / scale])

        print(f"Tag ID: {tag_id}, Offset: {offset}")

        # Extract relative rotation (yaw) from the detection
        rot = R.from_matrix(R_ca)

        z_rot = rot.as_euler('xyz', degrees=False)[1]

        # Rotate offset in the opposite direction of z_rot
        R_z_rot = np.array([
            [np.cos(-z_rot), -np.sin(-z_rot)],
            [np.sin(-z_rot),  np.cos(-z_rot)]
        ])
        offset = R_z_rot.dot(offset)
        offset[1] = -offset[1]

        # Get tag's world position and orientation
        tag_world_pos = np.array(self.tag_id_map[tag_id])
        tag_yaw_deg = self.tag_id_map_rot[tag_id][2]
        tag_yaw = tag_yaw_deg * (np.pi / 180.0)

        # Apply tag world rotation
        R_tag = np.array([
            [np.cos(tag_yaw), -np.sin(tag_yaw)],
            [np.sin(tag_yaw),  np.cos(tag_yaw)]
        ])
        robot_world = tag_world_pos + R_tag.dot(offset)

        # Adjust the robot's world yaw
        robot_world_yaw = tag_yaw  # Since we already corrected offset, this simplifies orientation

        return tag_id, offset, robot_world, z_rot, robot_world_yaw


    def run(self, ep_camera):
        while True:
            try:
                img = ep_camera.read_cv2_image(strategy="newest", timeout=0.5)
            except Empty:
                time.sleep(0.001)
                continue

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            detections = self.detector.detect(gray, estimate_tag_pose=True,
                                              camera_params=self.camera_params, tag_size=self.marker_size_m)
            for detection in detections:
                result = self.pw(detection)
                if result:
                    K = 1
                    vw = np.array([0,0])
                    vw = K * (np.array([4,11]) - self.pw(detection)[2]) 
                    # self.pdw_log.append(self.pdw())
                    # self.pw_log.append(self.pw(detection))
                    # self.timer +=0.5
                    # # self.write_to_log(f"TimeStep: {self.timer*2}")
                    # self.write_to_log(f"Current tag id: {self.id_of_last_detection}")
                    # self.write_to_log(f"vw: {vw}, pdw: {self.pdw()}, pw: {self.pw(detection)}, vdw: {self.vdw()}")
                    # print("vw:",vw,"pdw:",self.pdw(),"pw:",self.pw(detection),"vdw:",self.vdw())
                    t_ca, R_ca = self.get_pose_apriltag_in_camera_frame(detection)
                    rot = R.from_matrix(R_ca)
                   
                    z_rot = rot.as_euler('zyx',degrees=False)[0]
                    curr_rot = self.tag_id_map_rot[self.get_apriltag_id(detection)][2] * (np.pi / 180) + z_rot
                    print("Z_ROT VELOCITY",z_rot)
                    R_tag = np.array([
                        [np.cos(curr_rot), -np.sin(curr_rot)],
                        [np.sin(curr_rot),  np.cos(curr_rot)]
                    ])
                    print("VW1 BEFORE: ",vw)
                    vw1 = vw @ R_tag 
                    vw2 = np.dot(vw, R_tag)
                    print("VW1 After: ",vw1)
                    print("VW2 After: ",vw2)
                    tag_id, offset, robot_world, z_rot, robot_world_yaw = result
                    print(f"Tag ID {tag_id}:")
                    print(f"  Relative offset (in tag frame): {offset}")
                    print(f"  Robot World Position: {robot_world}")
                    print(f"  Detected relative rotation (rad): {z_rot}")
                    print(f"  Robot World Yaw (rad): {robot_world_yaw}")
                   # print(f" Calculated Velocity: {vw1}")
            time.sleep(1)

if __name__ == '__main__':
    np.set_printoptions(precision=3, suppress=True, linewidth=120)

    ep_robot = robot.Robot()
    ep_robot.initialize(conn_type="sta", sn="3JKCH8800100YN")
    ep_camera = ep_robot.camera
    ep_camera.start_video_stream(display=False, resolution=camera.STREAM_360P)

    K = np.array([[314, 0, 320], [0, 314, 180], [0, 0, 1]])
    marker_size_m = 0.153
    pw_test = PWTest(K, marker_size_m=marker_size_m)

    try:
        pw_test.run(ep_camera)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(traceback.format_exc())
    finally:
        print('Shutting down')
        ep_camera.stop_video_stream()
        ep_robot.close()
