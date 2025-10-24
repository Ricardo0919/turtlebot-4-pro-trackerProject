# Turtlebot 4 Pro - Person Tracker Project
# TurtleBot4 Pro — Person Follower (ROS 2 Humble + OAK‑D Pro + YOLOv8)

## Project Overview
This project enables a TurtleBot4 Pro to follow a person smoothly and reliably using its onboard OAK‑D Pro camera and ROS 2 Humble. Detection is powered by a YOLOv8 model trained in PyTorch and deployed to the camera via ONNX → OpenVINO blob, so inference runs on‑device while the Raspberry Pi focuses on control and messaging.

## Objectives
- Detect the `person` class in real time and obtain 3D position (X, Y, Z) from the OAK‑D Pro.
- Track and select a single target consistently, even in crowded scenes.
- Generate smooth velocity commands to follow at a comfortable distance.
- Provide clear interfaces (topics, messages) for easy integration with navigation or logging tools.
- Keep CPU usage on the robot low by offloading inference to the camera NPU.

## Hardware
- TurtleBot4 Pro (iRobot Create 3 base + Raspberry Pi 4)
- Luxonis OAK‑D Pro (RGB + stereo depth + IR projector)
- Development laptop (for model training and testing)
- Power and networking accessories as required

## Technology Stack
- **ROS 2 Humble (Ubuntu 22.04)** for middleware and lifecycle management.
- **OAK‑D Pro / DepthAI** for spatial AI and depth estimation.
- **YOLOv8 (Ultralytics, PyTorch)** for model training and evaluation.
- **ONNX + OpenVINO blob** for portable, efficient on‑device inference.
- **Custom ROS 2 node** (`person_follower`) for control and decision logic.

## System Architecture (High‑Level)
- **Perception Layer:** OAK‑D Pro runs the compiled neural network and publishes spatial detections (bounding box + 3D coordinates + confidence).
- **Tracking Layer:** Maintains the selected target over time to avoid ID swaps.
- **Control Layer:** Computes linear and angular velocities from distance and heading errors, applying filtering, limits, and safety rules.
- **Robot I/O:** Velocity commands are sent to the Create 3; sensor data and status are exposed via standard ROS 2 topics.

## Core Functionality
- Real‑time person detection with depth‑aware localization.
- Target selection and hand‑off management when multiple people appear.
- Smooth motion with bounded velocities, rate limiting, deadbands, and safety stops.
- Clear, documented topics for detections, diagnostics, and velocity commands.

## Control Strategy
- **Distance control:** PI controller to maintain a target following distance.
- **Heading control:** PD controller to keep the person centered in the field of view.
- **Stability aids:** Exponential smoothing, deadbands, anti‑windup, slew‑rate limits, and minimum‑distance safeguards.

## Data Flow (Conceptual)
1. OAK‑D Pro performs detection and depth estimation on‑device.
2. Spatial detections (with X, Y, Z and confidence) are published to ROS 2.
3. The follower node selects the target, computes errors, and outputs `/cmd_vel`.
4. The Create 3 executes motion while monitoring safety constraints.

## Expected Deliverables
- Documented model artifacts (PyTorch `.pt`, exported `.onnx`, and camera `.blob`).
- ROS 2 packages for camera bring‑up and the `person_follower` node.
- High‑level documentation of topics, parameters, and integration points.
- Demo videos and evaluation metrics (latency, FPS, tracking stability).
