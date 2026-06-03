#!/usr/bin/env python3
"""
Dual-Sensor Obstacle Avoidance v10
Auto-detects and fuses: Depth Camera (primary) + LIDAR (secondary) + RGB fallback

Features:
- Auto-discovers available sensors (depth image, lidar scan, rgb image)
- Depth camera: direct metric distance from depth pixels
- LIDAR: 360° horizontal scan for side/rear obstacles
- Sensor fusion: depth for front precision, lidar for peripheral awareness
- Track following with dynamic speed based on closest obstacle
- Proper body-frame velocity commands for PX4
"""

import asyncio
import math
import os
import threading
import time
from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional, List, Tuple, Dict
from collections import deque

import cv2
import numpy as np
import rclpy
from mavsdk import System
from mavsdk.offboard import OffboardError, VelocityBodyYawspeed
from rclpy.node import Node
from sensor_msgs.msg import Image, LaserScan


class State(Enum):
    CRUISE = auto()
    SLOW = auto()      # Obstacle detected, reduce speed
    AVOID = auto()     # Steer around obstacle
    BYPASS = auto()    # Lateral slide
    RECOVER = auto()   # Return to track


@dataclass
class Obstacle:
    direction: str      # "front_left", "front_center", "front_right", "left", "right", "rear"
    distance: float     # meters
    sensor: str         # "depth", "lidar", "rgb"
    timestamp: float


class DualSensorAvoidance(Node):
    # ---------- NAVIGATION ----------
    WAYPOINTS: List[Tuple[float, float]] = [
        (30.0, 0.0),
        (30.0, 30.0),
        (0.0, 30.0),
        (0.0, 0.0),
    ]
    WP_TOLERANCE_M = 3.0
    MAX_SPEED = 2.0
    MIN_SPEED = 0.4
    MAX_YAW_RATE = 30.0

    # ---------- DEPTH CAMERA ----------
    DEPTH_WIDTH = 320
    DEPTH_HEIGHT = 240
    DEPTH_MIN_M = 0.3       # Minimum valid depth
    DEPTH_MAX_M = 20.0      # Maximum valid depth
    DEPTH_STOP_M = 2.0      # Stop/avoid distance
    DEPTH_WARN_M = 5.0      # Slow down distance
    DEPTH_FOV_H = 90.0      # Horizontal FOV degrees
    DEPTH_FOV_V = 60.0      # Vertical FOV degrees

    # Depth regions (fractions of image width)
    REGION_LEFT = 0.35
    REGION_RIGHT = 0.65

    # Vertical crop: ignore ground (bottom 30%) and sky (top 10%)
    DEPTH_BOTTOM_CROP = 0.30
    DEPTH_TOP_CROP = 0.10

    # ---------- LIDAR ----------
    LIDAR_STOP_M = 1.5
    LIDAR_WARN_M = 3.0
    # Angular sectors (degrees from front)
    LIDAR_FRONT_LEFT = (-45, -15)
    LIDAR_FRONT_CENTER = (-15, 15)
    LIDAR_FRONT_RIGHT = (15, 45)
    LIDAR_LEFT = (-90, -45)
    LIDAR_RIGHT = (45, 90)
    LIDAR_REAR_LEFT = (-135, -90)
    LIDAR_REAR_RIGHT = (90, 135)

    # ---------- AVOIDANCE ----------
    AVOID_LATERAL_SPEED = 1.0
    AVOID_FORWARD_SPEED = 0.5
    BYPASS_DURATION_S = 2.5
    BYPASS_FORWARD = 0.3
    RECOVER_DURATION_S = 2.0
    OBSTACLE_MEMORY_S = 1.0

    def __init__(self):
        super().__init__("dual_sensor_avoidance_v10")

        # Sensor availability
        self.has_depth = False
        self.has_lidar = False
        self.has_rgb = False
        self.depth_topic = None
        self.lidar_topic = None
        self.rgb_topic = None

        # Discover sensors
        self._discover_sensors()

        # Image/scan storage
        self.latest_depth = None
        self.latest_rgb = None
        self.latest_scan = None
        self.frame_count = 0
        self.depth_count = 0
        self.lidar_count = 0

        # Optical flow fallback state
        self.prev_gray = None
        self.flow_features = dict(maxCorners=100, qualityLevel=0.2, minDistance=5, blockSize=5)
        self.flow_lk = dict(winSize=(15, 15), maxLevel=2,
                           criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

        # Navigation
        self.current_wp_idx = 0
        self.pos_x = 0.0
        self.pos_y = 0.0
        self.current_heading = 0.0
        self.state = State.CRUISE
        self.state_start_time = 0.0
        self.avoid_direction = 0.0
        self.airborne = False
        self.altitude = 5.0

        # Obstacles
        self.obstacles: Dict[str, Obstacle] = {}
        self.last_avoid_time = 0.0

        self.get_logger().info("Dual Sensor Avoidance v10 initialized")
        self.get_logger().info(f"  Depth: {self.has_depth} ({self.depth_topic or 'N/A'})")
        self.get_logger().info(f"  LIDAR: {self.has_lidar} ({self.lidar_topic or 'N/A'})")
        self.get_logger().info(f"  RGB:   {self.has_rgb} ({self.rgb_topic or 'N/A'})")

    def _discover_sensors(self):
        """Auto-discover depth camera, lidar, and RGB camera topics."""
        for attempt in range(30):
            topics = {n: t for n, t in self.get_topic_names_and_types()}

            # Look for depth camera
            for topic, types in topics.items():
                if "sensor_msgs/msg/Image" in types:
                    topic_lower = topic.lower()
                    if any(k in topic_lower for k in ["depth", "depth_camera", "depth_image"]):
                        if not self.has_depth:
                            self.depth_topic = topic
                            self.create_subscription(Image, topic, self._depth_cb, 10)
                            self.has_depth = True
                            self.get_logger().info(f"Depth camera: {topic}")
                    elif any(k in topic_lower for k in ["camera", "gimbal", "rgb", "image"]):
                        if not self.has_rgb and not any(k in topic_lower for k in ["depth"]):
                            self.rgb_topic = topic
                            self.create_subscription(Image, topic, self._rgb_cb, 10)
                            self.has_rgb = True
                            self.get_logger().info(f"RGB camera: {topic}")

            # Look for LIDAR
            for topic, types in topics.items():
                if "sensor_msgs/msg/LaserScan" in types:
                    if not self.has_lidar:
                        self.lidar_topic = topic
                        self.create_subscription(LaserScan, topic, self._lidar_cb, 10)
                        self.has_lidar = True
                        self.get_logger().info(f"LIDAR: {topic}")

            if self.has_depth or self.has_lidar or self.has_rgb:
                return

            time.sleep(1.0)
            rclpy.spin_once(self, timeout_sec=0.1)

        self.get_logger().error("No sensors found! Check ros_gz_bridge.")

    def _depth_cb(self, msg: Image):
        """Process depth image."""
        try:
            if msg.encoding == "32FC1":
                depth = np.frombuffer(msg.data, dtype=np.float32).reshape((msg.height, msg.width))
            elif msg.encoding == "16UC1":
                depth = np.frombuffer(msg.data, dtype=np.uint16).reshape((msg.height, msg.width)) / 1000.0
            else:
                return
        except:
            return

        self.latest_depth = depth
        self.depth_count += 1
        self.frame_count += 1

        if self.airborne and self.altitude > 2.0:
            self._process_depth(depth)

    def _process_depth(self, depth: np.ndarray):
        """Extract obstacles from depth image."""
        h, w = depth.shape
        now = time.time()

        # Resize for speed
        if w != self.DEPTH_WIDTH or h != self.DEPTH_HEIGHT:
            depth = cv2.resize(depth, (self.DEPTH_WIDTH, self.DEPTH_HEIGHT))
            h, w = self.DEPTH_HEIGHT, self.DEPTH_WIDTH

        # Crop ground and sky
        top = int(h * self.DEPTH_TOP_CROP)
        bottom = int(h * (1.0 - self.DEPTH_BOTTOM_CROP))
        depth_cropped = depth[top:bottom, :]

        # Filter valid depth values
        valid = (depth_cropped > self.DEPTH_MIN_M) & (depth_cropped < self.DEPTH_MAX_M)
        if not np.any(valid):
            return

        # Split into regions
        left_bound = int(w * self.REGION_LEFT)
        right_bound = int(w * self.REGION_RIGHT)

        regions = {
            "front_left": depth_cropped[:, :left_bound],
            "front_center": depth_cropped[:, left_bound:right_bound],
            "front_right": depth_cropped[:, right_bound:]
        }

        for name, reg in regions.items():
            reg_valid = valid[:, :left_bound] if name == "front_left" else                        valid[:, left_bound:right_bound] if name == "front_center" else                        valid[:, right_bound:]

            if not np.any(reg_valid):
                continue

            # Use minimum distance (closest obstacle) in region
            min_dist = np.min(reg[reg_valid])

            if min_dist < self.DEPTH_MAX_M:
                self.obstacles[name] = Obstacle(name, float(min_dist), "depth", now)

    def _lidar_cb(self, msg: LaserScan):
        """Process LIDAR scan."""
        self.latest_scan = msg
        self.lidar_count += 1

        if not self.airborne or self.altitude <= 2.0:
            return

        now = time.time()
        ranges = np.array(msg.ranges)
        angles = np.arange(msg.angle_min, msg.angle_max + msg.angle_increment/2, msg.angle_increment)

        # Filter valid ranges
        valid = (ranges > msg.range_min) & (ranges < msg.range_max) & ~np.isinf(ranges) & ~np.isnan(ranges)

        sectors = {
            "front_left": self.LIDAR_FRONT_LEFT,
            "front_center": self.LIDAR_FRONT_CENTER,
            "front_right": self.LIDAR_FRONT_RIGHT,
            "left": self.LIDAR_LEFT,
            "right": self.LIDAR_RIGHT,
            "rear_left": self.LIDAR_REAR_LEFT,
            "rear_right": self.LIDAR_REAR_RIGHT,
        }

        for name, (angle_min, angle_max) in sectors.items():
            # Convert to radians
            rad_min = math.radians(angle_min)
            rad_max = math.radians(angle_max)

            mask = (angles >= rad_min) & (angles < rad_max) & valid
            if not np.any(mask):
                continue

            sector_ranges = ranges[mask]
            min_dist = np.min(sector_ranges)

            # Only update if closer than existing or no existing
            if name not in self.obstacles or min_dist < self.obstacles[name].distance:
                self.obstacles[name] = Obstacle(name, float(min_dist), "lidar", now)

    def _rgb_cb(self, msg: Image):
        """RGB fallback using optical flow."""
        try:
            data = np.frombuffer(msg.data, dtype=np.uint8)
            if msg.encoding == "rgb8":
                frame = data.reshape((msg.height, msg.width, 3))[:, :, ::-1]
            elif msg.encoding == "bgr8":
                frame = data.reshape((msg.height, msg.width, 3))
            elif msg.encoding == "mono8":
                frame = cv2.cvtColor(data.reshape((msg.height, msg.width)), cv2.COLOR_GRAY2BGR)
            else:
                return
        except:
            return

        self.latest_rgb = frame
        self.frame_count += 1

        if self.airborne and self.altitude > 2.0 and not self.has_depth:
            self._process_rgb_flow(frame)

    def _process_rgb_flow(self, frame: np.ndarray):
        """Optical flow fallback when no depth camera."""
        small = cv2.resize(frame, (240, 160))
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        now = time.time()

        if self.prev_gray is None or self.prev_gray.shape != gray.shape:
            self.prev_gray = gray
            return

        mask = np.zeros_like(gray)
        mask[int(h*0.1):int(h*0.7), int(w*0.05):int(w*0.95)] = 255

        p0 = cv2.goodFeaturesToTrack(self.prev_gray, mask=mask, **self.flow_features)
        self.prev_gray = gray.copy()

        if p0 is None or len(p0) < 6:
            return

        p1, st, _ = cv2.calcOpticalFlowPyrLK(self.prev_gray, gray, p0, None, **self.flow_lk)
        if p1 is None or st is None:
            return

        p0 = p0.reshape(-1, 2)[st.reshape(-1) == 1]
        p1 = p1.reshape(-1, 2)[st.reshape(-1) == 1]

        if len(p0) < 6:
            return

        flow = p1 - p0
        mags = np.linalg.norm(flow, axis=1)
        mm = np.median(mags)

        if mm < 0.2:
            return

        valid = mags < mm * 4
        p0 = p0[valid]
        flow = flow[valid]
        mags = mags[valid]

        x = p0[:, 0]
        cx = w / 2

        for name, xmask in [("front_left", x < w*0.35),
                            ("front_center", (x >= w*0.35) & (x < w*0.65)),
                            ("front_right", x >= w*0.65)]:
            if np.any(xmask):
                fm = mags[xmask]
                if len(fm) >= 3 and np.mean(fm) > 0.5:
                    dist = max(0.5, 12.0 / np.mean(fm))
                    self.obstacles[name] = Obstacle(name, float(dist), "rgb", now)

    def _get_closest_obstacle(self) -> Optional[Obstacle]:
        """Get closest confirmed obstacle, expiring old ones."""
        now = time.time()
        # Expire old obstacles
        expired = [k for k, o in self.obstacles.items() if now - o.timestamp > self.OBSTACLE_MEMORY_S]
        for k in expired:
            del self.obstacles[k]

        if not self.obstacles:
            return None

        return min(self.obstacles.values(), key=lambda o: o.distance)

    def _get_front_obstacles(self) -> List[Obstacle]:
        """Get front-facing obstacles only."""
        front_dirs = {"front_left", "front_center", "front_right"}
        return [o for o in self.obstacles.values() if o.direction in front_dirs]

    def update_state(self, heading: float, x: float, y: float, alt: float):
        now = time.time()
        self.current_heading = heading
        self.pos_x = x
        self.pos_y = y
        self.altitude = alt

        closest = self._get_closest_obstacle()
        front_obs = self._get_front_obstacles()
        closest_front = min(front_obs, key=lambda o: o.distance) if front_obs else None

        # Track bearing
        wx, wy = self._get_target_wp()
        dx = wx - x
        dy = wy - y
        dist_to_wp = math.hypot(dx, dy)
        bearing = math.degrees(math.atan2(dy, dx))
        hdg_err = heading_error(bearing, heading)

        # WP arrival
        if dist_to_wp < self.WP_TOLERANCE_M and self.state == State.CRUISE:
            self._advance_wp()
            wx, wy = self._get_target_wp()
            dx = wx - x
            dy = wy - y
            dist_to_wp = math.hypot(dx, dy)
            bearing = math.degrees(math.atan2(dy, dx))
            hdg_err = heading_error(bearing, heading)

        # Determine threat level
        threat = False
        threat_dist = 999.0
        threat_dir = "none"

        if closest_front and (now - self.last_avoid_time) > 1.5:
            threat_dist = closest_front.distance
            threat_dir = closest_front.direction
            if threat_dist < self.DEPTH_STOP_M:
                threat = True
            elif threat_dist < self.DEPTH_WARN_M:
                threat = True

        # Also check side lidar for very close obstacles
        if not threat and closest and closest.sensor == "lidar" and closest.distance < self.LIDAR_STOP_M:
            if closest.direction in {"left", "right"}:
                threat = True
                threat_dist = closest.distance
                threat_dir = closest.direction

        # ---------- STATE MACHINE ----------
        if self.state == State.CRUISE:
            if threat:
                self.state = State.AVOID
                self.state_start_time = now

                # Pick avoid direction
                if "left" in threat_dir:
                    self.avoid_direction = 1.0   # go right
                elif "right" in threat_dir:
                    self.avoid_direction = -1.0  # go left
                else:
                    self.avoid_direction = 1.0 if hdg_err > 0 else -1.0

                self.get_logger().warn(f"AVOID: {threat_dir} @ {threat_dist:.1f}m ({closest_front.sensor if closest_front else 'unknown'})")
            else:
                # Speed based on clearance
                speed = self.MAX_SPEED
                if closest_front:
                    d = closest_front.distance
                    if d < self.DEPTH_WARN_M:
                        speed = self.MIN_SPEED + (self.MAX_SPEED - self.MIN_SPEED) * (d / self.DEPTH_WARN_M)

                yaw_rate = float(np.clip(hdg_err * 1.5, -self.MAX_YAW_RATE, self.MAX_YAW_RATE))
                return speed, 0.0, 0.0, yaw_rate

        elif self.state == State.AVOID:
            if not threat or (closest_front and closest_front.distance > self.DEPTH_WARN_M * 1.2):
                self.state = State.CRUISE
                self.get_logger().info("→ CRUISE (clear)")
            elif now - self.state_start_time > 1.0:
                self.state = State.BYPASS
                self.state_start_time = now
                self.get_logger().info("→ BYPASS")
            else:
                # Slow forward + lateral + turn away
                yaw_rate = self.avoid_direction * 25.0
                lateral = self.avoid_direction * self.AVOID_LATERAL_SPEED * 0.5
                return self.AVOID_FORWARD_SPEED, lateral, 0.0, yaw_rate

        elif self.state == State.BYPASS:
            elapsed = now - self.state_start_time

            if elapsed > self.BYPASS_DURATION_S:
                if threat and closest_front and closest_front.distance < self.DEPTH_STOP_M * 1.5:
                    self.state_start_time = now  # extend
                    self.get_logger().warn("BYPASS extended")
                else:
                    self.state = State.RECOVER
                    self.state_start_time = now
                    self.last_avoid_time = now
                    self.get_logger().info("→ RECOVER")

            yaw_rate = self.avoid_direction * 15.0
            lateral = self.avoid_direction * self.AVOID_LATERAL_SPEED
            return self.BYPASS_FORWARD, lateral, 0.0, yaw_rate

        elif self.state == State.RECOVER:
            elapsed = now - self.state_start_time

            if elapsed > self.RECOVER_DURATION_S:
                self.state = State.CRUISE
                self.get_logger().info("→ CRUISE")

            yaw_rate = float(np.clip(hdg_err * 2.0, -self.MAX_YAW_RATE, self.MAX_YAW_RATE))
            return self.MAX_SPEED * 0.7, 0.0, 0.0, yaw_rate

        return 0.0, 0.0, 0.0, 0.0

    def _advance_wp(self):
        self.current_wp_idx = (self.current_wp_idx + 1) % len(self.WAYPOINTS)
        self.get_logger().info(f"WP → #{self.current_wp_idx} {self._get_target_wp()}")

    def _get_target_wp(self):
        return self.WAYPOINTS[self.current_wp_idx]

    def get_vis(self):
        if self.latest_rgb is not None:
            vis = self.latest_rgb.copy()
        elif self.latest_depth is not None:
            # Normalize depth for visualization
            d = self.latest_depth.copy()
            dvis = np.clip((d / 10.0) * 255, 0, 255).astype(np.uint8)
            dvis = cv2.applyColorMap(dvis, cv2.COLORMAP_JET)
            vis = cv2.resize(dvis, (640, 480))
        else:
            return None

        h, w = vis.shape[:2]
        cv2.rectangle(vis, (0, 0), (w, 150), (0, 0, 0), -1)

        colors = {State.CRUISE: (0, 255, 0), State.SLOW: (0, 255, 255),
                  State.AVOID: (0, 165, 255), State.BYPASS: (0, 0, 255),
                  State.RECOVER: (255, 0, 255)}
        c = colors.get(self.state, (128, 128, 128))

        wp = self._get_target_wp()
        cv2.putText(vis, f"V10 {self.state.name} → WP{self.current_wp_idx}({wp[0]:.0f},{wp[1]:.0f})",
                   (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, c, 2)

        sensors = []
        if self.has_depth: sensors.append("DEPTH")
        if self.has_lidar: sensors.append("LIDAR")
        if self.has_rgb: sensors.append("RGB")
        cv2.putText(vis, f"Sensors: {'+'.join(sensors)}",
                   (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        closest = self._get_closest_obstacle()
        if closest:
            cv2.putText(vis, f"CLOSEST: {closest.direction} {closest.distance:.1f}m ({closest.sensor})",
                       (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        # List all obstacles
        y_off = 90
        for obs in sorted(self.obstacles.values(), key=lambda o: o.distance)[:3]:
            cv2.putText(vis, f"  {obs.direction}: {obs.distance:.1f}m ({obs.sensor})",
                       (10, y_off), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)
            y_off += 15

        cv2.putText(vis, f"Pos:({self.pos_x:.1f},{self.pos_y:.1f}) Hdg:{self.current_heading:.0f}° Alt:{self.altitude:.1f}m",
                   (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)

        return vis

    def set_airborne(self, a, alt=0.0):
        self.airborne = a
        self.altitude = alt


def heading_error(t, c):
    t, c = t % 360, c % 360
    d = t - c
    if d > 180: d -= 360
    elif d < -180: d += 360
    return d


def spin_ros2(n):
    rclpy.spin(n)


async def run():
    rclpy.init()
    node = DualSensorAvoidance()
    threading.Thread(target=spin_ros2, args=(node,), daemon=True).start()

    drone = System()
    await drone.connect(system_address="udp://:14540")

    print("=" * 60)
    print("DUAL-SENSOR OBSTACLE AVOIDANCE v10")
    print(f"Track: {node.WAYPOINTS}")
    print(f"Depth: {node.has_depth} | LIDAR: {node.has_lidar} | RGB: {node.has_rgb}")
    print("=" * 60)

    print("\n[1] GPS...")
    async for h in drone.telemetry.health():
        if h.is_global_position_ok:
            print("    OK")
            break

    print("[2] Sensors...")
    for _ in range(60):
        if node.depth_count > 3 or node.lidar_count > 3 or node.frame_count > 5:
            print(f"    OK (D:{node.depth_count} L:{node.lidar_count} R:{node.frame_count})")
            break
        await asyncio.sleep(1.0)

    print("[3] Offboard...")
    await drone.offboard.set_velocity_body(VelocityBodyYawspeed(0.0, 0.0, 0.0, 0.0))
    try:
        await drone.offboard.start()
    except OffboardError as e:
        print(f"    FAIL: {e._result.result}")
        return

    print("[4] Takeoff to 5m...")
    await drone.action.arm()
    await asyncio.sleep(2)
    await drone.offboard.set_velocity_body(VelocityBodyYawspeed(0.0, 0.0, -2.0, 0.0))
    await asyncio.sleep(5)
    await drone.offboard.set_velocity_body(VelocityBodyYawspeed(0.0, 0.0, 0.0, 0.0))
    await asyncio.sleep(2)

    node.set_airborne(True, 5.0)
    print("[5] AIRBORNE — Dual sensor avoidance active")

    vis_path = os.path.expanduser("~/PX4-Autopilot/dual_sensor_v10.jpg")

    try:
        while True:
            hdg = 0.0
            alt = 5.0
            pos_x = 0.0
            pos_y = 0.0

            async for h in drone.telemetry.heading():
                hdg = h.heading_deg
                break
            async for p in drone.telemetry.position():
                alt = p.relative_altitude_m
                break
            try:
                async for odom in drone.telemetry.odometry():
                    if odom.position_body:
                        pos_x = odom.position_body.x_m
                        pos_y = odom.position_body.y_m
                    break
            except:
                pass

            fwd, lat, vert, yaw = node.update_state(hdg, pos_x, pos_y, alt)

            vert += -(5.0 - alt) * 0.5
            vert = float(np.clip(vert, -2.0, 1.0))

            await drone.offboard.set_velocity_body(VelocityBodyYawspeed(fwd, lat, vert, yaw))

            closest = node._get_closest_obstacle()
            obs_str = ""
            if closest:
                obs_str = f" | CLOSEST:{closest.direction} {closest.distance:.1f}m"

            line = (f"\rState: {node.state.name:8s} | F:{fwd:5.1f} L:{lat:5.1f} V:{vert:5.1f} Y:{yaw:6.1f} | "
                   f"Pos:({pos_x:.1f},{pos_y:.1f}) | Alt:{alt:.1f}m{obs_str}")
            print(line, end="", flush=True)

            if (node.depth_count + node.lidar_count + node.frame_count) % 30 == 0:
                v = node.get_vis()
                if v is not None:
                    cv2.imwrite(vis_path, v)

            await asyncio.sleep(0.1)

    except KeyboardInterrupt:
        print("\n\nInterrupted")

    print("[6] Land...")
    node.set_airborne(False)
    await drone.offboard.set_velocity_body(VelocityBodyYawspeed(0.0, 0.0, 1.0, 0.0))
    await asyncio.sleep(5)
    try:
        await drone.offboard.stop()
    except:
        pass
    await drone.action.disarm()
    node.destroy_node()
    rclpy.shutdown()
    print("Done")


if __name__ == "__main__":
    asyncio.run(run())

