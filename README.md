# uav-digital-interface-px4-ros2
# UAV Digital Interface System — PX4 + ROS 2 + Gazebo Harmonic

**Engineer:** Ahmed Bouzid  
**Status:** Phase 1 — Operational (May 2026)  
**Stack:** PX4 Autopilot · ROS 2 Jazzy · Gazebo Harmonic · 
MAVSDK-Python · OpenCV · Ubuntu 24.04 LTS

## What This Project Does
End-to-end autonomous UAV simulation stack with:
- Multi-waypoint autonomous mission execution
- Real-time telemetry streaming at 100 Hz
- Camera sensor integration + aerial photography
- Optical flow obstacle avoidance (Lucas-Kanade)
- Full PX4 SITL + QGroundControl integration

## Key Achievements
- 98% real-time simulation factor on consumer hardware
- 40+ integration issues resolved and documented
- Autonomous missions: 150m → 300m → 500m waypoints
- Optical flow detection at ~25 FPS

## Tech Stack
| Component | Version |
|---|---|
| PX4 Autopilot | v1.15.x |
| ROS 2 | Jazzy Jalisco |
| Gazebo | Harmonic LTS |
| MAVSDK-Python | Latest |
| OpenCV | 4.x |
| Ubuntu | 24.04 LTS |

## Full Technical Report
See `docs/UAV_Digital_Interface_Report_v1.5.pdf`
