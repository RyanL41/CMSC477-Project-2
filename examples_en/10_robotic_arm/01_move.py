# -*-coding:utf-8-*-
# Copyright (c) 2020 DJI.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License in the file LICENSE.txt or at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import robomaster
from robomaster import robot


if __name__ == "__main__":
    ep_robot = robot.Robot()
    ep_robot.initialize(conn_type="sta", sn="3JKCH8800100YN")

    ep_arm = ep_robot.robotic_arm

    # Move forward 20mm
    # ep_arm.move(x=20, y=0).wait_for_completed()
    # Move backward 20mm
    # ep_arm.move(x=-20, y=0).wait_for_completed()
    # Move upward 20mm
    ep_arm.move(x=0, y=20).wait_for_completed()
    # Move downward 20mm
    # ep_arm.move(x=0, y=-20).wait_for_completed()

    ep_robot.close()
