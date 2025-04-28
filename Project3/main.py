"""
Main entry point for Project 3 - RoboMaster EP robot control.
"""
import traceback
from Project3.state_machine import RobotStateMachine

def main():
    # Robot serial number and map file
    robot_sn = "3JKCH8800100YN"
    map_file = "./InitialMap.csv"
    
    # Create and run the state machine
    state_machine = RobotStateMachine(robot_sn, map_file)
    
    try:
        state_machine.run()
    except KeyboardInterrupt:
        print("\nProgram interrupted by user")
    except Exception as e:
        print(f"Error: {e}")
        print(traceback.format_exc())
    finally:
        print("Shutting down robot...")


if __name__ == "__main__":
    main()
