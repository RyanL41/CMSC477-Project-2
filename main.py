"""
Main entry point for P3-Revamp - RoboMaster EP robot control system.
"""
import time
import traceback
import sys
import os

# Add the parent directory to sys.path to enable imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from P3_Revamp.state_machine import RobotStateMachine
from P3_Revamp.utilities import log_info


def main():
    """Main entry point for the program."""
    # Hardcoded robot serial number
    robot_sn = "3JKCH8800100YN"
    
    # Debug mode flag
    debug = True  # Set to True for debug output
    
    # Print startup information
    log_info("=== P3-Revamp: RoboMaster EP Robot Control System ===")
    log_info(f"Robot SN: {robot_sn}")
    log_info(f"Debug Mode: {'Enabled' if debug else 'Disabled'}")
    log_info("Starting in 3 seconds...")
    time.sleep(3)
    
    # Create and run the state machine
    state_machine = RobotStateMachine(robot_sn, debug=debug)
    
    try:
        state_machine.run()
    except KeyboardInterrupt:
        log_info("\nProgram interrupted by user")
    except Exception as e:
        log_info(f"Error: {e}")
        traceback.print_exc()
    finally:
        log_info("Shutting down robot...")


if __name__ == "__main__":
    main()