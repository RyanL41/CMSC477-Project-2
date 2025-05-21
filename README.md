# P3-Revamp: RoboMaster EP Control System

This is a revamped implementation of Project 3 for the RoboMaster EP robot. It uses a new approach to navigation and task execution, focusing on target-oriented movement, vision-based object detection, and AprilTag-based localization.

## Features

- **AprilTag-Based Localization**: Uses AprilTags for accurate position tracking
- **Target-Oriented Path Planning**: Direct movement toward targets, no grid-based planning
- **Vision-Based Object Detection**: YOLO-based detection of Lego blocks, center line, and closet
- **Thread-Safe Architecture**: Separate threads for vision processing and movement
- **State Machine Control**: Clear state transitions for complex task sequences
- **Obstacle Avoidance**: Detect and avoid AprilTags and other robots

## System Components

- **Localization**: Tracks robot position using AprilTags
- **Path Planning**: Navigates to target positions while searching for objects
- **Lego Pickup**: Centers, approaches, and picks up Lego blocks
- **Lego Dropoff**: Finds the closet and deposits Lego blocks
- **Center Line Handling**: Detects and centers on the center line

## Installation

1. Ensure you have the RoboMaster SDK installed:

   ```bash
   pip install robomaster
   ```

2. Install required dependencies:

   ```bash
   pip install opencv-python numpy pupil-apriltags ultralytics
   ```

3. Make sure you're connected to the same network as the RoboMaster EP robot.

## Usage

Run the main program:

```bash
python3 -m main
```

## Task Sequence

The robot follows this sequence of actions:

1. **Localization**: Find AprilTags to establish position
2. **Path Planning (Lego)**: Move to position (10, -4) while looking for Lego blocks
3. **Lego Pickup**: Center on, approach, and pick up Lego block
4. **Path Planning (Center Line Pre)**: Move to position (6, -3) while looking for center line
5. **Intermediary Centering**: Rotate to 270° and center on center line
6. **Path Planning (Closet)**: Move forward while looking for closet
7. **Lego Dropoff**: Approach closet and drop off Lego block
8. **Path Planning (Center Line Post)**: Return to saved position
9. **Path Planning (Return)**: Return to starting position

This sequence repeats, allowing the robot to continuously collect and deliver Lego blocks.

## Architecture

- **Modular Design**: Each component focuses on a specific task
- **Thread Separation**: Vision processing never issues movement commands
- **State Machine**: Coordinates components and manages transitions
- **Configuration-Driven**: Parameters defined in central config file

## Customization

You can adjust parameters in `config.py`:

- Target positions
- Detection thresholds
- Movement speeds
- Robot starting position

## Troubleshooting

- Make sure the robot is connected to the same network as your computer
- Check that the robot serial number is correct
- Ensure the camera is working properly
- Verify that YOLO model weights are correctly placed

## Credits

This project is a revamped implementation based on ideas from the original Project 3, with a new approach to navigation and control.

## License

[Add appropriate license information here]
