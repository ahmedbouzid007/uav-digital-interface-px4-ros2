#!/usr/bin/env python3
"""
Unified 3D LiDAR + Depth Camera Obstacle Avoidance
PX4 + Gazebo Harmonic + ROS 2 Jazzy

Fuses:
- Depth Camera: high-rate frontal precision (front_left, front_center, front_right)
- 3D LiDAR: 360° awareness with vertical cropping (all sectors)

Features:
- Auto-discovers depth and 3D LiDAR topics
- EMA smoothing per sector for both sensors
- Depth takes priority for front sectors when fresh
- Dynamic react distances based on current speed
- Ramped velocity outputs for smooth PX4 offboard control
"""

import asyncio
import math
import os
import threading
import time
from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional, List, Tuple, Dict

import cv2
import numpy as np
import rclpy
from mavsdk import System
from mavsdk.offboard import OffboardError, VelocityBodyYawspeed
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image, PointCloud2
import sensor_msgs_py.point_cloud2 as pc2


class State(Enum):
    CRUISE = auto()
    AVOID = auto()
    BYPASS = auto()
    RECOVER = auto()


@dataclass
class Obstacle:
    direction: str
    distance: float
    sensor: str
    timestamp: float


class UnifiedAvoidance(Node):
    # ---------- NAVIGATION ----------
    WAYPOINTS: List[Tuple[float, float]] = [
        (40.0, 0.0), (40.0, 40.0), (0.0, 40.0), (0.0, 0.0),
    ]
    WP_TOLERANCE_M = 3.0
    MAX_SPEED = 1.8
    MIN_SPEED = 0.2
    CRUISE_YAW_RATE = 25.0
    SPEED_RAMP_RATE = 0.4
    YAW_RAMP_RATE = 12.0
    LATERAL_RAMP_RATE = 0.35

    # ---------- DEPTH CAMERA ----------
    DEPTH_WIDTH = 320
    DEPTH_HEIGHT = 240
    DEPTH_MIN_M = 0.3
    DEPTH_MAX_M = 20.0
    DEPTH_STOP_M = 2.0
    DEPTH_WARN_M = 5.0
    REGION_LEFT = 0.35
    REGION_RIGHT = 0.65
    DEPTH_BOTTOM_CROP = 0.30
    DEPTH_TOP_CROP = 0.10
    DEPTH_EMA_ALPHA = 0.5

    # ---------- 3D LIDAR ----------
    LIDAR_MIN_M = 0.3
    LIDAR_MAX_M = 30.0
    VERTICAL_FOV_CROP = math.radians(25.0)
    LIDAR_EMA_ALPHA = 0.4
    OBSTACLE_MEMORY_S = 2.0

    SECTORS = {
        "front_left": (-60, -15),
        "front_center": (-15, 15),
        "front_right": (15, 60),
        "left": (-100, -60),
        "right": (60, 100),
        "rear_left": (-135, -100),
        "rear_right": (100, 135),
    }

    # ---------- AVOIDANCE ----------
    AVOID_LATERAL_SPEED = 1.2
    AVOID_FORWARD_SPEED = 0.3
    BYPASS_DURATION_S = 2.5
    BYPASS_FORWARD = 0.2
    RECOVER_DURATION_S = 1.5

    def __init__(self):
        super().__init__("unified_avoidance_3d_lidar_depth")

        # Sensor state
        self.has_depth = False
        self.has_lidar = False
        self.depth_topic = None
        self.lidar_topic = None

        # Data storage
        self.latest_depth = None
        self.lidar_count = 0
        self.depth_count = 0
        self.last_lidar_time = 0.0

        # Navigation state
        self.current_wp_idx = 0
        self.pos_x = 0.0
        self.pos_y = 0.0
        self.current_heading = 0.0
        self.state = State.CRUISE
        self.state_start_time = 0.0
        self.avoid_direction = 0.0
        self.airborne = False
        self.altitude = 5.0

        # Ramped outputs (smooth control)
        self.current_speed = 0.0
        self.current_yaw_rate = 0.0
        self.current_lateral = 0.0

        # Obstacles
        self.obstacles: Dict[str, Obstacle] = {}
        self.obstacle_ema: Dict[str, float] = {}
        self.last_avoid_time = 0.0

        # Discover and subscribe
        self._discover_sensors()

        self.get_logger().info("=" * 50)
        self.get_logger().info("Unified 3D LiDAR + Depth Avoidance initialized")
        self.get_logger().info(f"  Depth:    {self.has_depth} ({self.depth_topic or 'N/A'})")
        self.get_logger().info(f"  3D LiDAR: {self.has_lidar} ({self.lidar_topic or 'N/A'})")
        self.get_logger().info("=" * 50)

    def _discover_sensors(self):
        """Auto-discover depth camera and 3D LiDAR topics."""
        for attempt in range(30):
            topics = {n: t for n, t in self.get_topic_names_and_types()}

            # Depth camera (Image with depth/depth_camera/depth_image in name)
            for topic, types in topics.items():
                if "sensor_msgs/msg/Image" in types:
                    tl = topic.lower()
                    if any(k in tl for k in ["depth", "depth_camera", "depth_image"]):
                        if not self.has_depth:
                            self.depth_topic = topic
                            self.create_subscription(Image, topic, self._depth_cb, 10)
                            self.has_depth = True
                            self.get_logger().info(f"Depth camera: {topic}")

            # 3D LiDAR (PointCloud2)
            for topic, types in topics.items():
                if "sensor_msgs/msg/PointCloud2" in types:
                    if not self.has_lidar:
                        self.lidar_topic = topic
                        self.create_subscription(PointCloud2, topic, self._lidar_cb, qos_profile_sensor_data)
                        self.has_lidar = True
                        self.get_logger().info(f"3D LiDAR: {topic}")

            if self.has_depth or self.has_lidar:
                return

            time.sleep(1.0)
            rclpy.spin_once(self, timeout_sec=0.1)

        self.get_logger().error("No sensors found! Check ros_gz_bridge topics.")

    def _depth_cb(self, msg: Image):
        """Process depth image (32FC1 or 16UC1)."""
        try:
            if msg.encoding == "32FC1":
                depth = np.frombuffer(msg.data, dtype=np.float32).reshape((msg.height, msg.width))
            elif msg.encoding == "16UC1":
                depth = np.frombuffer(msg.data, dtype=np.uint16).reshape((msg.height, msg.width)) / 1000.0
            else:
                return
        except Exception:
            return

        self.latest_depth = depth
        self.depth_count += 1

        if self.airborne and self.altitude > 2.0:
            self._process_depth(depth)

    def _process_depth(self, depth: np.ndarray):
        """Extract frontal obstacles from depth image."""
        h, w = depth.shape
        now = time.time()

        # Resize for consistent processing speed
        if w != self.DEPTH_WIDTH or h != self.DEPTH_HEIGHT:
            depth = cv2.resize(depth, (self.DEPTH_WIDTH, self.DEPTH_HEIGHT))
            h, w = self.DEPTH_HEIGHT, self.DEPTH_WIDTH

        # Crop ground (bottom 30%) and sky (top 10%)
        top = int(h * self.DEPTH_TOP_CROP)
        bottom = int(h * (1.0 - self.DEPTH_BOTTOM_CROP))
        depth_cropped = depth[top:bottom, :]

        # Filter valid depth values
        valid = (depth_cropped > self.DEPTH_MIN_M) & (depth_cropped < self.DEPTH_MAX_M)
        if not np.any(valid):
            return

        left_bound = int(w * self.REGION_LEFT)
        right_bound = int(w * self.REGION_RIGHT)

        regions = {
            "front_left": (depth_cropped[:, :left_bound], valid[:, :left_bound]),
            "front_center": (depth_cropped[:, left_bound:right_bound], valid[:, left_bound:right_bound]),
            "front_right": (depth_cropped[:, right_bound:], valid[:, right_bound:]),
        }

        for name, (reg, reg_valid) in regions.items():
            if not np.any(reg_valid):
                continue

            min_dist = float(np.min(reg[reg_valid]))

            # EMA smoothing
            if name in self.obstacle_ema:
                self.obstacle_ema[name] = (
                    self.DEPTH_EMA_ALPHA * min_dist +
                    (1 - self.DEPTH_EMA_ALPHA) * self.obstacle_ema[name]
                )
            else:
                self.obstacle_ema[name] = min_dist

            self.obstacles[name] = Obstacle(name, self.obstacle_ema[name], "depth", now)

    def _lidar_cb(self, msg: PointCloud2):
        """Process 3D LiDAR point cloud."""
        self.lidar_count += 1
        self.last_lidar_time = time.time()

        try:
            points = list(pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True))
            if len(points) == 0:
                return

            # Handle both structured arrays and plain tuples
            pts = np.array(points)
            if pts.dtype.names is not None:
                x = pts['x'].astype(np.float32)
                y = pts['y'].astype(np.float32)
                z = pts['z'].astype(np.float32)
            else:
                pts = pts.astype(np.float32)
                x = pts[:, 0]
                y = pts[:, 1]
                z = pts[:, 2]

            ranges = np.sqrt(x*x + y*y + z*z)
            azimuth = np.arctan2(y, x)
            horizontal_dist = np.sqrt(x*x + y*y)
            elevation = np.arctan2(z, horizontal_dist)

            # Filter by range and vertical angle (ignore ground/sky)
            valid = (
                (ranges > self.LIDAR_MIN_M) &
                (ranges < self.LIDAR_MAX_M) &
                (np.abs(elevation) < self.VERTICAL_FOV_CROP)
            )

            if not np.any(valid):
                return

            valid_az = azimuth[valid]
            valid_ranges = ranges[valid]
            now = time.time()

            for name, (deg_min, deg_max) in self.SECTORS.items():
                rad_min = math.radians(deg_min)
                rad_max = math.radians(deg_max)

                if rad_min > rad_max:
                    sector_mask = (valid_az >= rad_min) | (valid_az < rad_max)
                else:
                    sector_mask = (valid_az >= rad_min) & (valid_az < rad_max)

                if not np.any(sector_mask):
                    continue

                sector_ranges = valid_ranges[sector_mask]
                dist = float(np.percentile(sector_ranges, 10))

                # EMA smoothing
                if name in self.obstacle_ema:
                    self.obstacle_ema[name] = (
                        self.LIDAR_EMA_ALPHA * dist +
                        (1 - self.LIDAR_EMA_ALPHA) * self.obstacle_ema[name]
                    )
                else:
                    self.obstacle_ema[name] = dist

                # FUSION: Depth takes priority for front sectors when fresh (<0.5s)
                if name in {"front_left", "front_center", "front_right"}:
                    if name in self.obstacles and self.obstacles[name].sensor == "depth":
                        depth_age = now - self.obstacles[name].timestamp
                        if depth_age < 0.5:
                            continue

                self.obstacles[name] = Obstacle(name, self.obstacle_ema[name], "lidar", now)

            if self.lidar_count % 20 == 0:
                fl = self.obstacle_ema.get("front_left", float('inf'))
                fc = self.obstacle_ema.get("front_center", float('inf'))
                fr = self.obstacle_ema.get("front_right", float('inf'))
                self.get_logger().info(
                    f"LiDAR #{self.lidar_count}: pts={len(pts)}, FL={fl:.1f}, FC={fc:.1f}, FR={fr:.1f}"
                )

        except Exception as e:
            self.get_logger().error(f"3D LiDAR callback error: {e}")

    def _expire_obstacles(self):
        """Remove stale obstacles."""
        now = time.time()
        expired = []
        for k, o in list(self.obstacles.items()):
            if now - o.timestamp > self.OBSTACLE_MEMORY_S:
                expired.append(k)
        for k in expired:
            del self.obstacles[k]
            if k in self.obstacle_ema:
                del self.obstacle_ema[k]

    def _get_front_obstacles(self) -> List[Obstacle]:
        self._expire_obstacles()
        front_dirs = {"front_left", "front_center", "front_right"}
        return [o for o in self.obstacles.values() if o.direction in front_dirs]

    def _get_all_obstacles(self) -> List[Obstacle]:
        self._expire_obstacles()
        return list(self.obstacles.values())

    def _get_closest(self, obstacles: List[Obstacle]) -> Optional[Obstacle]:
        if not obstacles:
            return None
        return min(obstacles, key=lambda o: o.distance)

    def _ramp_value(self, current: float, target: float, max_delta: float) -> float:
        """Smoothly ramp a value to avoid jerky control."""
        delta = target - current
        if abs(delta) <= max_delta:
            return target
        return current + math.copysign(max_delta, delta)

    def update_state(self, heading: float, x: float, y: float, alt: float):
        now = time.time()
        self.current_heading = heading
        self.pos_x = x
        self.pos_y = y
        self.altitude = alt

        front_obs = self._get_front_obstacles()
        closest_front = self._get_closest(front_obs)
        all_obs = self._get_all_obstacles()

        # Waypoint logic
        wx, wy = self.WAYPOINTS[self.current_wp_idx]
        dx = wx - x
        dy = wy - y
        dist_to_wp = math.hypot(dx, dy)
        bearing = math.degrees(math.atan2(dy, dx))
        hdg_err = heading_error(bearing, heading)

        if dist_to_wp < self.WP_TOLERANCE_M and self.state == State.CRUISE:
            self._advance_wp()
            wx, wy = self.WAYPOINTS[self.current_wp_idx]
            dx = wx - x
            dy = wy - y
            dist_to_wp = math.hypot(dx, dy)
            bearing = math.degrees(math.atan2(dy, dx))
            hdg_err = heading_error(bearing, heading)

        # Dynamic react distances based on current speed
        react_dist = self.current_speed * 2.5
        stop_dist = max(1.5, react_dist * 0.5)
        warn_dist = max(3.0, react_dist * 1.0)

        threat = False
        threat_dist = 999.0
        threat_dir = "none"

        if closest_front and (now - self.last_avoid_time) > 1.0:
            threat_dist = closest_front.distance
            threat_dir = closest_front.direction
            if threat_dist < stop_dist:
                threat = True
            elif threat_dist < warn_dist:
                threat = True

        # Side/rear emergency from LiDAR
        closest_all = self._get_closest(all_obs)
        if not threat and closest_all and closest_all.sensor == "lidar":
            if closest_all.distance < 1.2 and closest_all.direction in {"left", "right", "rear_left", "rear_right"}:
                threat = True
                threat_dist = closest_all.distance
                threat_dir = closest_all.direction

        target_speed = 0.0
        target_lateral = 0.0
        target_yaw_rate = 0.0

        emergency = False
        if closest_front and closest_front.distance < 1.0:
            emergency = True
            target_speed = -0.3

        # ---------- STATE MACHINE ----------
        if self.state == State.CRUISE:
            if emergency:
                self.state = State.AVOID
                self.state_start_time = now
                self.avoid_direction = 1.0 if hdg_err > 0 else -1.0
                self.get_logger().error(f"EMERGENCY: {threat_dir} @ {closest_front.distance:.1f}m")
            elif threat:
                self.state = State.AVOID
                self.state_start_time = now
                if "left" in threat_dir:
                    self.avoid_direction = 1.0
                elif "right" in threat_dir:
                    self.avoid_direction = -1.0
                else:
                    # front_center: pick side with more clearance
                    fl = self.obstacle_ema.get("front_left", float('inf'))
                    fr = self.obstacle_ema.get("front_right", float('inf'))
                    self.avoid_direction = 1.0 if fl > fr else -1.0
                self.get_logger().warn(f"AVOID: {threat_dir} @ {threat_dist:.1f}m")
            else:
                target_speed = self.MAX_SPEED
                if abs(hdg_err) > 45.0:
                    target_speed = self.MIN_SPEED
                elif abs(hdg_err) > 15.0:
                    target_speed = self.MIN_SPEED + (self.MAX_SPEED - self.MIN_SPEED) * ((45.0 - abs(hdg_err)) / 30.0)

                if closest_front:
                    d = closest_front.distance
                    if d < warn_dist:
                        target_speed = min(target_speed, self.MIN_SPEED + (self.MAX_SPEED - self.MIN_SPEED) * (d / warn_dist))

                target_yaw_rate = float(np.clip(hdg_err * 2.5, -self.CRUISE_YAW_RATE, self.CRUISE_YAW_RATE))

        elif self.state == State.AVOID:
            if not threat or (closest_front and closest_front.distance > warn_dist * 1.2):
                self.state = State.CRUISE
                self.get_logger().info("-> CRUISE (clear)")
            elif now - self.state_start_time > 0.8:
                self.state = State.BYPASS
                self.state_start_time = now
                self.get_logger().info("-> BYPASS")
            else:
                target_speed = self.AVOID_FORWARD_SPEED if not emergency else -0.3
                target_lateral = self.avoid_direction * self.AVOID_LATERAL_SPEED * 0.5
                target_yaw_rate = self.avoid_direction * 45.0

        elif self.state == State.BYPASS:
            elapsed = now - self.state_start_time
            if elapsed > self.BYPASS_DURATION_S:
                if threat and closest_front and closest_front.distance < stop_dist * 1.5:
                    self.state_start_time = now
                    self.get_logger().warn("BYPASS extended")
                else:
                    self.state = State.RECOVER
                    self.state_start_time = now
                    self.last_avoid_time = now
                    self.get_logger().info("-> RECOVER")

            progress = elapsed / self.BYPASS_DURATION_S
            target_speed = self.BYPASS_FORWARD + (self.MAX_SPEED * 0.5 - self.BYPASS_FORWARD) * progress
            target_lateral = self.avoid_direction * self.AVOID_LATERAL_SPEED * (1.0 - progress * 0.3)
            yaw_toward_track = float(np.clip(hdg_err * 2.0, -30.0, 30.0))
            target_yaw_rate = self.avoid_direction * 25.0 + yaw_toward_track

        elif self.state == State.RECOVER:
            elapsed = now - self.state_start_time
            if elapsed > self.RECOVER_DURATION_S:
                self.state = State.CRUISE
                self.get_logger().info("-> CRUISE")
            target_speed = self.MAX_SPEED * 0.6
            target_yaw_rate = float(np.clip(hdg_err * 3.0, -45.0, 45.0))

        # Ramp outputs for smooth PX4 control
        self.current_speed = self._ramp_value(self.current_speed, target_speed, self.SPEED_RAMP_RATE)
        self.current_lateral = self._ramp_value(self.current_lateral, target_lateral, self.LATERAL_RAMP_RATE)
        self.current_yaw_rate = self._ramp_value(self.current_yaw_rate, target_yaw_rate, self.YAW_RAMP_RATE)

        return self.current_speed, self.current_lateral, 0.0, self.current_yaw_rate

    def _advance_wp(self):
        self.current_wp_idx = (self.current_wp_idx + 1) % len(self.WAYPOINTS)
        self.get_logger().info(f"WP -> #{self.current_wp_idx} {self.WAYPOINTS[self.current_wp_idx]}")

    def set_airborne(self, a, alt=0.0):
        self.airborne = a
        self.altitude = alt

    def get_vis(self):
        """Generate visualization from depth image."""
        if self.latest_depth is not None:
            d = self.latest_depth.copy()
            dvis = np.clip((d / 10.0) * 255, 0, 255).astype(np.uint8)
            vis = cv2.applyColorMap(dvis, cv2.COLORMAP_JET)
            vis = cv2.resize(vis, (640, 480))
        else:
            return None

        h, w = vis.shape[:2]
        cv2.rectangle(vis, (0, 0), (w, 150), (0, 0, 0), -1)

        colors = {
            State.CRUISE: (0, 255, 0),
            State.AVOID: (0, 165, 255),
            State.BYPASS: (0, 0, 255),
            State.RECOVER: (255, 0, 255),
        }
        c = colors.get(self.state, (128, 128, 128))

        wp = self.WAYPOINTS[self.current_wp_idx]
        cv2.putText(vis, f"Unified {self.state.name} -> WP{self.current_wp_idx}({wp[0]:.0f},{wp[1]:.0f})",
                   (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, c, 2)

        sensors = []
        if self.has_depth: sensors.append("DEPTH")
        if self.has_lidar: sensors.append("3D-LIDAR")
        cv2.putText(vis, f"Sensors: {'+'.join(sensors)}",
                   (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        closest = self._get_closest(self._get_all_obstacles())
        if closest:
            cv2.putText(vis, f"CLOSEST: {closest.direction} {closest.distance:.1f}m ({closest.sensor})",
                       (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        y_off = 90
        for obs in sorted(self.obstacles.values(), key=lambda o: o.distance)[:5]:
            cv2.putText(vis, f"  {obs.direction}: {obs.distance:.1f}m ({obs.sensor})",
                       (10, y_off), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)
            y_off += 15

        cv2.putText(vis, f"Pos:({self.pos_x:.1f},{self.pos_y:.1f}) Hdg:{self.current_heading:.0f} Alt:{self.altitude:.1f}m",
                   (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)

        return vis


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
    node = UnifiedAvoidance()
    threading.Thread(target=spin_ros2, args=(node,), daemon=True).start()

    drone = System()
    await drone.connect(system_address="udp://:14540")

    print("=" * 60)
    print("UNIFIED 3D LIDAR + DEPTH OBSTACLE AVOIDANCE")
    print(f"Track: {node.WAYPOINTS}")
    print(f"Depth: {node.has_depth} | 3D LiDAR: {node.has_lidar}")
    print("=" * 60)

    print("\n[1] GPS...")
    async for h in drone.telemetry.health():
        if h.is_global_position_ok:
            print("    OK")
            break

    print("[2] Waiting for sensors...")
    for _ in range(60):
        if node.depth_count > 3 or node.lidar_count > 5:
            print(f"    OK (D:{node.depth_count} L:{node.lidar_count})")
            break
        await asyncio.sleep(1.0)
    if node.depth_count <= 3 and node.lidar_count <= 5:
        print("    WARNING: Limited sensor data")

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
    print("[5] AIRBORNE - Unified avoidance active")

    vis_path = os.path.expanduser("~/PX4-Autopilot/unified_avoidance.jpg")

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

            if alt < 1.0 and node.airborne:
                alt = 5.0

            fwd, lat, vert, yaw = node.update_state(hdg, pos_x, pos_y, alt)
            vert += -(5.0 - alt) * 0.5
            vert = float(np.clip(vert, -2.0, 1.0))

            await drone.offboard.set_velocity_body(VelocityBodyYawspeed(fwd, lat, vert, yaw))

            fl = node.obstacle_ema.get("front_left", float('inf'))
            fc = node.obstacle_ema.get("front_center", float('inf'))
            fr = node.obstacle_ema.get("front_right", float('inf'))
            fl_s = f"{fl:.1f}" if fl < 99 else "--"
            fc_s = f"{fc:.1f}" if fc < 99 else "--"
            fr_s = f"{fr:.1f}" if fr < 99 else "--"

            lidar_age = time.time() - node.last_lidar_time
            lidar_health = "OK" if lidar_age < 1.0 else f"STALE({lidar_age:.1f}s)"

            closest = node._get_closest(node._get_all_obstacles())
            obs_str = f" | {closest.direction}:{closest.distance:.1f}m" if closest else ""

            line = (f"\rState:{node.state.name:8s} | F:{fwd:5.2f} L:{lat:5.2f} | "
                   f"FL:{fl_s} FC:{fc_s} FR:{fr_s} | LIDAR:{lidar_health} | "
                   f"Pos:({pos_x:.1f},{pos_y:.1f}){obs_str}")
            print(line, end="", flush=True)

            if (node.depth_count + node.lidar_count) % 30 == 0:
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
