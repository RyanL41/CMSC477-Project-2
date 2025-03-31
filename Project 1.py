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

from djikstra import get_path

class DjikstraStates:
    LOOKING_FOR_START_TAG = "looking_for_start_tag"
    LOOKING_FOR_TAG = "looking_for_tag"
    PARALLEL_TO_TAG = "parallel_to_tag"
    GOING_TO_START_POSITION = "going_to_start_position"
    FOLLOWING_PATH = "following_path"
    LOOKING_FOR_TAG_ON_PATH = "looking_for_tag_on_path"
    COMPLETED = "completed"

class VelocityControl:
    def __init__(self, csv_path):
        self.csv_path = csv_path
        self.curr_state = DjikstraStates.LOOKING_FOR_START_TAG
        self.filename = "logs"
        self.log_folder = "logs-folder"
        self.clear_and_create_log()
        os.makedirs(self.log_folder, exist_ok=True)

        self.positions = get_path(csv_path)
        self.timer = 0
        self.sleeptime = 0
        self.id_of_last_detection = None
        self.speed_limit = 0.2
        
        self.tag_id_map = {
            30: (2, 2.5),
            31: (2, 3.5),
            32: (4, 2.5),
            33: (4, 3.5),
            34: (5.5, 3),
            35: (1.5, 5),
            36: (1.5, 7),
            37: (4.5, 6),
            38: (6, 5.5),
            39: (6, 6.5),
            40: (8, 5.5),
            41: (8, 6.5),
            42: (2, 8.5),
            43: (2, 9.5),
            44: (4, 8.5),
            45: (4, 9.5),
            46: (5.5, 9),
        }

        self.tag_id_map_rot = { # rotation ccw
            30: (2, 2.5, 0),
            31: (2, 3.5, 180),
            32: (4, 2.5, 0),
            33: (4, 3.5, 180),
            34: (5.5, 3, 90),
            35: (1.5, 5, 90),
            36: (1.5, 7, 90),
            37: (4.5, 6, -90), #originally 270?
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

        self.x_vel = 0
        self.y_vel = 0
        self.rot_vel = 10

        self.pw_log = []  # Store pw values over time
        self.pdw_log = []  # Store pdw values over time
        self.time_log = []  # Store timestamps

    def get_interpolated_path(self, positions):
        return

    def pdw(self):
        
            
        pdw_value = self.positions[min(int(self.timer), len(self.positions)-1)]
        #self.pdw_log.append(pdw_value)
        
        return np.array(pdw_value) # just to make sure

    def get_apriltag_id(self, detection):
        return detection.tag_id


    def pw(self, detection):
        """
        Computes the robot's position (and overall yaw) in the world frame, adjusting for orientation
        to make it "parallel" to the detected AprilTag.
        """
        tag_id = get_apriltag_id(detection)

        if tag_id not in self.tag_id_map or tag_id not in self.tag_id_map_rot:
            print("NO TAG ID IN MAP")
            return None

        # Get tag pose in camera frame
        t_ca, R_ca = get_pose_apriltag_in_camera_frame(detection)
        scale = 0.266  # scaling factor from image units to meters

        # Compute initial offset
        offset = np.array([-t_ca[0] / scale, t_ca[2] / scale])

        #print(f"Tag ID: {tag_id}, Offset: {offset}")

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

        return robot_world


    def log_pw_pdw_charts(self):
        time_steps = list(range(len(self.pw_log)))
        pw_x = [p[0] for p in self.pw_log]
        pw_y = [p[1] for p in self.pw_log]
        pdw_x = [p[0] for p in self.pdw_log]
        pdw_y = [p[1] for p in self.pdw_log]
        
        plt.figure()
        plt.plot(time_steps, pw_x, label='pw_x')
        plt.plot(time_steps, pdw_x, label='pdw_x')
        plt.xlabel('Time Step')
        plt.ylabel('Position')
        plt.title('pw_x Over Time')
        plt.legend()
        plt.savefig(os.path.join(self.log_folder, 'pw_plot.png'))
        plt.close()
        
        plt.figure()
        plt.plot(time_steps, pw_y, label='pw_y')
        plt.plot(time_steps, pdw_y, label='pdw_y')
        plt.xlabel('Time Step')
        plt.ylabel('Position')
        plt.title('pw_y Over Time')
        plt.legend()
        plt.savefig(os.path.join(self.log_folder, 'pdw_plot.png'))
        plt.close()

    def vdw(self):
        #NEED TO MODIFIY time step to be accurate to how much time will pass between desired positions (seconds)
        time_step = 5
        return (self.positions[min(int(self.timer)+1,len(self.positions)-1)] - self.positions[min(int(self.timer),len(self.positions)-1)])/time_step 
    
    def get_last_detection(self, detections, parallel=False):

        # Yeah, I think Levi was mentioning some distortion in the camera's corners - we should probably
        # use t_ca[0] as a threshold. What do you thinK

        #That could work, I was also think we just grab the detection with the smallest difference between pw and pdw
        
        # ok sounds good, let's do your idea first

        #K
        #Do we need to modify this function in order to achieve that?

        # I think we do - let me check

        # yeah, we shouldn't check id_of_last_detection anymore, we just need to get pdw and do a lowest distance check

        # Sounds good though I am worried we lose the ability to rotate to watch the target. Not sure if this will be a problem
        
        # yeah excited to see what happens through vscode comments lol

        #valid_detection = [d for d in detections if get_apriltag_id(d) == self.id_of_last_detection]
        valid_detection = None
        min_ = np.array([10000,10000])   # min is keyword :O
        for i in detections:
            
            #print("difference:",(self.pdw() - self.pw(i)),"Norm",np.linalg.norm((self.pdw() - self.pw(i))),"Min",np.linalg.norm(min_),"valid_detection",valid_detection)
            #Modifiy last number for lower tolerance 
            if np.linalg.norm((self.pdw() - self.pw(i))) < np.linalg.norm(min_): 
            #and self.get_apriltag_id(i) != 37:
                min_ = self.pdw() - self.pw(i) 
                valid_detection = i

                
        #if not valid_detection: return None
        if valid_detection == None:
        
            return None
        
        #detection = valid_detection[0]

        # if parallel or True: 
        #     return detection

        _, R_ca = get_pose_apriltag_in_camera_frame(valid_detection)
        rot = R.from_matrix(R_ca)
        z_rot = rot.as_euler('zyx',degrees=True)[1]
        #This line was always returning true
        #if abs(z_rot) < 30: return None
        #print("HERE")
        return valid_detection

    def get_rotation_to_aruco_tag(self, detection):
        """
        If the x_pos is positive, rotate clockwise. Else, rotate counterclockwise. 
        """
        t_ca, _ = get_pose_apriltag_in_camera_frame(detection)
        x_pos = t_ca[0]
#Ok gonna try test now
        rotation = max(-0.3, min(x_pos, 0.3)) * 100

        self.write_to_log(f"Going to rotate {rotation} since x_pos is {x_pos}")

        return rotation

    def get_degree(self, detection):
        _, R_ca = get_pose_apriltag_in_camera_frame(detection)
        rot = R.from_matrix(R_ca)
        z_rot = rot.as_euler('zyx',degrees=True)[1]
        return z_rot

    def write_to_log(self, text):
        """Appends the given text to the log file."""
        with open(self.filename, 'a') as log_file:
            log_file.write(text + '\n')

    def clear_and_create_log(self):
        """Clears the existing log file or creates a new one."""
        with open(self.filename, 'w') as log_file:
            # Optionally, add an initial header or message
            log_file.write("Log file created/cleared.\n")

    def run(self):
        while True:
            try:
                img = ep_camera.read_cv2_image(strategy="newest", timeout=0.5)
            except Empty:
                time.sleep(0.001)
                continue

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray.astype(np.uint8)

            detections = apriltag.find_tags(gray)
            
            if self.curr_state == DjikstraStates.LOOKING_FOR_START_TAG:
                self.rot_vel
                if len(detections) > 0:
                    for i in detections:
                        if get_apriltag_id(i) == 32:
                            self.curr_state = DjikstraStates.PARALLEL_TO_TAG
                            # stop rotation
                            self.rot_vel = 0 

            elif self.curr_state == DjikstraStates.LOOKING_FOR_TAG:

                self.rot_vel = 10

                if len(detections) > 0:

                    for detection in detections:
                        _, R_ca = get_pose_apriltag_in_camera_frame(detection)
                        rot = R.from_matrix(R_ca)
                        z_rot = rot.as_euler('zyx',degrees=True)[1]

                        if abs(z_rot) < 25:

                            self.id_of_last_detection = get_apriltag_id(detections[0])
                            
                            self.curr_state = DjikstraStates.PARALLEL_TO_TAG
                            # stop rotation
                            self.rot_vel = 0 
                    
            elif self.curr_state == DjikstraStates.LOOKING_FOR_TAG_ON_PATH:

                self.rot_vel = 10
                useful_detections = []
                for i in detections:
                    self.write_to_log(f"Tagid:{self.get_apriltag_id(i)} Norm {np.linalg.norm((self.pdw() - self.pw(i)))} PDW {self.pdw()}")
                if len(detections) > 0:
                    useful_detections = [d for d in detections if np.linalg.norm((self.pdw() - self.pw(d))) < 2]

                    
                if len(useful_detections) > 0:

                    for detection in useful_detections:
                        _, R_ca = get_pose_apriltag_in_camera_frame(detection)
                        rot = R.from_matrix(R_ca)
                        z_rot = rot.as_euler('zyx',degrees=True)[1]

                        if np.linalg.norm((self.pdw() - self.pw(detection))) < 2:

                            self.id_of_last_detection = get_apriltag_id(detections[0])
                            
                            self.curr_state = DjikstraStates.FOLLOWING_PATH
                            # stop rotation
                            self.rot_vel = 0 
                
            elif self.curr_state == DjikstraStates.PARALLEL_TO_TAG:

                detection = self.get_last_detection(detections, parallel=True)
                #print("Detection:",detection)
                if not detection: 

                    self.curr_state = DjikstraStates.LOOKING_FOR_TAG
                
                else:
                    
                    t_ca, R_ca = get_pose_apriltag_in_camera_frame(detection)
                    rot = R.from_matrix(R_ca)
                    z_rot = rot.as_euler('zyx',degrees=True)[1]
                    print(z_rot)
                    if z_rot < -15:
                        self.rot_vel = -15
                    elif z_rot > 15:
                        self.rot_vel = 15
                    else:
                        self.rot_vel = 0
                        self.curr_state = DjikstraStates.GOING_TO_START_POSITION

            elif self.curr_state == DjikstraStates.GOING_TO_START_POSITION:
                for i in detections:
                    april_tag_id = get_apriltag_id(i)
                    if april_tag_id == 32:
                        t_ca, R_ca = get_pose_apriltag_in_camera_frame(i)
                        k = 1
                        vel_y = k * (t_ca[0])
                        vel_x = k * (t_ca[2]-0.52)
                        # if vel_y < 0.1 and vel_y > -0.1 and vel_x < 0.1 and vel_x > -0.1:
                        if (t_ca[0] > 0.05 or t_ca[0] < -.05) or (t_ca[2] > 0.53 or t_ca[2] < .49):
                            self.x_vel = vel_x
                            self.y_vel = vel_y
                        else:
                            self.curr_state = DjikstraStates.FOLLOWING_PATH
                            self.x_vel = 0
                            self.y_vel = 0
                            self.rot_vel = 0

            elif self.curr_state == DjikstraStates.FOLLOWING_PATH:

                self.write_to_log("FOLLOWING PATH")

                self.x_vel = 0
                self.y_vel = 0
                self.rot_vel = 0
                detection = self.get_last_detection(detections)
                
                if detection:
                    print("Tag_Id:",self.get_apriltag_id(detection))
                    K = .5
                    vw = K * (self.pdw() - self.pw(detection)) + self.vdw()
                    self.pdw_log.append(self.pdw())
                    self.pw_log.append(self.pw(detection))
                    self.timer +=0.5
                    self.write_to_log(f"TimeStep: {self.timer*2}")
                    self.write_to_log(f"Current tag id: {self.id_of_last_detection}")
                    self.write_to_log(f"vw: {vw}, pdw: {self.pdw()}, pw: {self.pw(detection)}, vdw: {self.vdw()}")
                    # print("vw:",vw,"pdw:",self.pdw(),"pw:",self.pw(detection),"vdw:",self.vdw())
                    t_ca, R_ca = get_pose_apriltag_in_camera_frame(detection)
                    rot = R.from_matrix(R_ca)
                   
                    z_rot = rot.as_euler('zyx',degrees=False)[2]
                    
                    self.id_of_last_detection = self.get_apriltag_id(detection)
                    curr_rot = self.tag_id_map_rot[self.id_of_last_detection][2] * (np.pi / 180) + z_rot
                    rotation_matrix = np.array([
                    [np.cos(curr_rot), -np.sin(curr_rot)],
                    [np.sin(curr_rot), np.cos(curr_rot)]
                    ])
                    vw = vw @ rotation_matrix
                    self.id_of_last_detection = self.get_apriltag_id(detection)
                    curr_rot = self.tag_id_map_rot[self.id_of_last_detection][2] * (np.pi / 180) + z_rot
                    # self.y_vel = vw[0] * np.cos(curr_rot) - vw[1] * np.sin(curr_rot)
                    # self.x_vel = vw[0] * np.sin(curr_rot) + vw[1] * np.cos(curr_rot)
                    # #self.y_vel =0
                    self.x_vel = vw[1]
                    self.y_vel = vw[0]
                    self.write_to_log(f"x_vel: {self.x_vel}, y_vel: {self.y_vel}, curr_rot: {np.rad2deg(curr_rot)}")
                    self.rot_vel = self.get_rotation_to_aruco_tag(detection)

                    #self.x_vel, self.y_vel, self.rot_vel = 0, 0, 0

                else:
                    self.write_to_log("SWITCHING TAG ID")
                    self.curr_state = DjikstraStates.LOOKING_FOR_TAG_ON_PATH
                if abs(self.x_vel) > self.speed_limit:
                    if self.x_vel > 0:
                        self.x_vel = self.speed_limit
                    else:
                        self.x_vel = -self.speed_limit
                if abs(self.y_vel) > self.speed_limit:
                    if self.y_vel > 0:
                        self.y_vel = self.speed_limit
                    else:
                        self.y_vel = -self.speed_limit    
            print(self.curr_state)
            ep_chassis.drive_speed(x=self.x_vel, y=self.y_vel, z=self.rot_vel)
            #time.sleep(self.sleeptime)
            
            draw_detections(img, detections)
            cv2.imshow("img", img)
            if cv2.waitKey(1) == ord('q'):
                break
        self.log_pw_pdw_charts()  # Generate charts on exit


class AprilTagDetector:
    def __init__(self, K, family="tag36h11", threads=2, marker_size_m=0.16):
        self.camera_params = [K[0, 0], K[1, 1], K[0, 2], K[1, 2]]
        self.marker_size_m = marker_size_m
        self.detector = pupil_apriltags.Detector(family, threads)

    def find_tags(self, frame_gray):
        detections = self.detector.detect(frame_gray, estimate_tag_pose=True,
            camera_params=self.camera_params, tag_size=self.marker_size_m)
        return detections

def get_pose_apriltag_in_camera_frame(detection):
    R_ca = detection.pose_R
    t_ca = detection.pose_t
    tagid = detection.tag_id
    
    
    return t_ca.flatten(), R_ca
    
def get_apriltag_id(detection):
    tagid = detection.tag_id
    return tagid

def draw_detections(frame, detections):
    for detection in detections:
        pts = detection.corners.reshape((-1, 1, 2)).astype(np.int32)

        frame = cv2.polylines(frame, [pts], isClosed=True, color=(0, 0, 255), thickness=2)

        top_left = tuple(pts[0][0])  # First corner
        top_right = tuple(pts[1][0])  # Second corner
        bottom_right = tuple(pts[2][0])  # Third corner
        bottom_left = tuple(pts[3][0])  # Fourth corner
        cv2.line(frame, top_left, bottom_right, color=(0, 0, 255), thickness=2)
        cv2.line(frame, top_right, bottom_left, color=(0, 0, 255), thickness=2)

def detect_tag_loop(ep_robot, ep_chassis, ep_camera, apriltag):
    while True:
        try:
            img = ep_camera.read_cv2_image(strategy="newest", timeout=0.5)
        except Empty:
            time.sleep(0.001)
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray.astype(np.uint8)

        detections = apriltag.find_tags(gray)

        if len(detections) > 0:
            #assert len(detections) == 1 # Assume there is only one AprilTag to track
            detection = detections[0]

            t_ca, R_ca, tagid = get_pose_apriltag_in_camera_frame(detection)

            print(tagid)
            #print('R_ca', R_ca)
        #     #xyz
        #     k = 1
        #     vel_x = k * (t_ca[0])
        #     vel_y = k * (t_ca[2]-0.7)

        #     #rotation
        #     #beta = np.arcsin(-R_ca[2][0]) 
        #     #alpha = np.arccos(R_ca[0][0]/np.cos(beta)) 
            
        #     #if alpha > np.pi:
        #      #   alpha = -alpha
        #     rot = R.from_matrix(R_ca)
        #     z_rot = rot.as_euler('zyx',degrees=True)[0]
        #     print("z_rot",-z_rot)
        #     turn = 40
        #     if z_rot < -2:
        #         turn = 40
        #     elif z_rot > 2:
        #         turn = -40
        #     else:
        #         turn = 0

        #     ep_chassis.drive_speed(x=vel_y, y=vel_x, z=turn)
        # else:
        #     ep_chassis.drive_speed(x=0, y=0, z=0)
        draw_detections(img, detections)
        cv2.imshow("img", img)
        if cv2.waitKey(1) == ord('q'):
            break

if __name__ == '__main__':
    # More legible printing from numpy.
    np.set_printoptions(precision=3, suppress=True, linewidth=120)

    ep_robot = robot.Robot()
    ep_robot.initialize(conn_type="sta", sn="3JKCH8800100YN")#(conn_type="sta", sn="3JKCH7T00100J0")
    ep_chassis = ep_robot.chassis
    ep_camera = ep_robot.camera
    ep_camera.start_video_stream(display=False, resolution=camera.STREAM_360P)

    K = np.array([[314, 0, 320], [0, 314, 180], [0, 0, 1]]) # Camera focal length and center pixel
    marker_size_m = 0.153 # Size of the AprilTag in meters
    apriltag = AprilTagDetector(K, threads=2, marker_size_m=marker_size_m)

    state_machine = VelocityControl('./Project1Map.csv')

    print(state_machine.positions[0])

    try:
        state_machine.run()
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(traceback.format_exc())
    finally:
        print('Waiting for robomaster shutdown')
        ep_camera.stop_video_stream()
        ep_robot.close()
