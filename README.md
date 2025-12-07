# TurtleBot4 Pro — Person Tracker (ROS 2 Jazzy + OAK-D Pro + YOLOv8)

## Project Overview
This project implements a real-time 2D person-following system for the TurtleBot4 Pro using ROS 2 Jazzy, an OAK-D Pro RGB camera, and a YOLOv8 model exported to ONNX.  
Inference does **not** run on the OAK-D or on the TurtleBot’s Raspberry Pi, as both lack the required compute power. Instead, an external computer performs ONNX inference and sends results back to the robot through a dedicated high-speed Wi-Fi router (up to 3 Gb/s).  
The system tracks a person in image space (no depth/3D estimation) and sends velocity commands to make the robot follow the detected target.

## Objectives
- Detect the `person` class in real time using a YOLOv8 ONNX model.
- Track the target in **2D only** (horizontal position + bounding-box scale).
- Generate smooth `/cmd_vel` commands via a simple proportional controller.
- Maintain a lightweight ROS 2 architecture suitable for limited hardware.
- Achieve low latency by offloading all inference to an external machine.

## Hardware
- TurtleBot4 Pro (Create 3 base + Raspberry Pi 4)
- Luxonis OAK-D Pro (RGB only; depth not used)
- External computer (model inference + ROS 2 nodes)
- High-speed Wi-Fi router (≈ 3 Gb/s throughput for camera → PC → robot loop)

## Technology Stack
- **ROS 2 Jazzy (Ubuntu 24.04)**
- **YOLOv8 (Ultralytics)** trained in PyTorch, exported to **ONNX**
- **OpenCV + cv_bridge** for image processing
- **Custom ROS 2 nodes**:
  - `TrackerNodeOnnx` / `TrackerNodeScale` for 2D detection + tracking
  - `MotorsNode` for velocity control using a P-controller

## System Architecture (Actual Implementation)
- **Perception:**  
  The external computer receives RGB frames and performs ONNX inference to obtain:
  - normalized center `(x, y)`
  - bounding-box scale (`size_ratio`)
- **Tracking:**  
  The system selects the **highest-confidence** person detection (no identity tracking).
- **Control:**  
  A proportional yaw controller + threshold-based forward motion:
  - Rotate to center the person in the image  
  - Move forward until size_ratio reaches a target threshold
- **Robot I/O:**  
  Final `/cmd_vel` is sent to the Create 3.

_No depth, no 3D coordinates, no onboard inference._

## Control Strategy
### Heading Control (Yaw)
- Pure **P-controller**:
  ```
  omega = kp * (0.5 - nx)
  ```
- Includes deadband and saturation.

### Distance Control
- Based solely on bounding-box size:
  ```
  if size_ratio < far_threshold:
      vx = linear_speed
  else:
      vx = 0.0
  ```

## Data Flow
1. OAK-D Pro publishes RGB frames → external PC.
2. PC runs ONNX inference and selects best detection.
3. PC publishes:
   - `/tracking_person/target`
   - `/tracking_person/size_ratio`
   - `/tracking_person/annotated` (optional)
4. `MotorsNode` computes velocities and sends `/cmd_vel` to the robot.
5. The TurtleBot follows the detected person.

## Current Limitations
- No identity tracking → cannot guarantee it follows the same person in crowded scenes.
- No depth or spatial (3D) estimation.
- System depends on external compute + router for low latency.
- TurtleBot’s Raspberry Pi cannot run YOLO inference reliably.

## Future Improvements
- Add single-person filtering or re-identification to maintain target identity.
- Integrate stereo depth or custom distance estimation for 3D following.
- Move inference onboard using:
  - Intel NUC / Jetson Orin / Coral TPU, or
  - OAK-D OpenVINO blob (if optimized).
- Improve wireless link with stronger antennas or Wi-Fi 6 router.

## Drive Folder
[Google Drive Link](https://drive.google.com/drive/folders/1CY6xymP5jvlAziYTM0oSAsX0YRQBL3cf?usp=sharing)
