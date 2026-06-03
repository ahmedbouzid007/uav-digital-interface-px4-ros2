#!/usr/bin/env python3
"""
Dual-Sensor Obstacle Avoidance v12.3 — QoS FIX
- Switched sensor subscriptions to qos_profile_sensor_data (best effort)
  so they match ros_gz_bridge publishers in Gazebo Harmonic + ROS 2 Jazzy.
- Added LIDAR heartbeat monitor.
- Removed altitude gating that disabled processing.
- Faster state machine + emergency backup.
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
from sensor_msgs.msg import Image, LaserScan


class State(Enum):
    CRUISE = auto()
    SLOW = auto()
    AVOID = auto()
    BYPASS = auto()
    RECOVER = auto()


@dataclass
class Obstacle:
    direction: str
    distance: float
    sensor: str
    timestamp: float


class DualSensorAvoidanceV12(Node):
    # ---------- NAVIGATION ----------
    WAYPOINTS: List[Tuple[float, float]] = [
        (40.0, 0.0),
        (40.0, 40.0),
        (0.0, 40.0),
        (0.0, 0.0),
    ]
    WP_TOLERANCE_M = 3.0
    MAX_SPEED = 1.5
    MIN_SPEED = 0.2
    CRUISE_YAW_RATE = 20.0
    MAX_YAW_RATE = 30.0

    SPEED_RAMP_RATE = 0.4
    YAW_RAMP_RATE = 10.0

    # ---------- DEPTH CAMERA ----------
    DEPTH_WIDTH = 320
    DEPTH_HEIGHT = 240
    DEPTH_MIN_M = 0.5
    DEPTH_MAX_M = 25.0

    DEPTH_BOTTOM_CROP = 0.40
    DEPTH_TOP_CROP = 0.08
    DEPTH_EDGE_CROP = 0.10

    REGION_LEFT = 0.35
    REGION_RIGHT = 0.65

    REACT_TIME_S = 2.5

    # ---------- LIDAR ----------
    LIDAR_MIN_M = 0.2
    LIDAR_MAX_M = 20.0

    SECTORS = {
        "front_left": (-60, -20),
        "front_center": (-20, 20),
        "front_right": (20, 60),
        "left": (-100, -60),
        "right": (60, 100),
        "rear_left": (-160, -100),
        "rear_right": (100, 160),
    }

    # ---------- AVOIDANCE ----------
    AVOID_LATERAL_SPEED = 1.2
    AVOID_FORWARD_SPEED = 0.3
    BYPASS_DURATION_S = 2.0
    BYPASS_FORWARD = 0.2
    RECOVER_DURATION_S = 1.5
    OBSTACLE_MEMORY_S = 2.0

    DISTANCE_EMA_ALPHA = 0.4

    def __init__(self):
        super().__init__("dual_sensor_avoidance_v12")

        self.has_depth = False
        self.has_lidar = False
        self.has_rgb = False
        self.depth_topic = None
        self.lidar_topic = None
        self.rgb_topic = None

        self._discover_sensors()

        # Storage
        self.latest_depth = None
        self.latest_rgb = None
        self.latest_scan = None
        self.frame_count = 0
        self.depth_count = 0
        self.lidar_count = 0
        self.last_lidar_time = 0.0

        # RGB fallback
        self.prev_gray = None
        self.flow_features = dict(maxCorners=80, qualityLevel=0.25, minDistance=7, blockSize=5)
        self.flow_lk = dict(winSize=(15, 15), maxLevel=2,
                           criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

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

        # Smooth velocity tracking
        self.current_speed = 0.0
        self.current_yaw_rate = 0.0
        self.current_lateral = 0.0

        # Obstacles with EMA
        self.obstacles: Dict[str, Obstacle] = {}
        self.obstacle_ema: Dict[str, float] = {}
        self.last_avoid_time = 0.0

        self.get_logger().info("Dual Sensor Avoidance v12.3 initialized")
        self.get_logger().info(f"  Depth: {self.has_depth} ({self.depth_topic or 'N/A'})")
        self.get_logger().info(f"  LIDAR: {self.has_lidar} ({self.lidar_topic or 'N/A'})")
        self.get_logger().info(f"  RGB:   {self.has_rgb} ({self.rgb_topic or 'N/A'})")

    def _discover_sensors(self):
        for attempt in range(30):
            topics = {n: t for n, t in self.get_topic_names_and_types()}

            # --- DEPTH CAMERA ---
            depth_candidates = []
            for topic, types in topics.items():
                if "sensor_msgs/msg/Image" in types:
                    tl = topic.lower()
                    if any(k in tl for k in ["depth", "depth_camera", "depth_image", "distance"]):
                        if not any(k in tl for k in ["rgb", "color", "left", "right"]):
                            depth_candidates.append(topic)

            if depth_candidates and not self.has_depth:
                self.depth_topic = min(depth_candidates, key=len)
                self.create_subscription(Image, self.depth_topic, self._depth_cb, qos_profile_sensor_data)
                self.has_depth = True
                self.get_logger().info(f"Depth camera: {self.depth_topic}")

            # --- LIDAR ---
            lidar_candidates = []
            for topic, types in topics.items():
                if "sensor_msgs/msg/LaserScan" in types:
                    tl = topic.lower()
                    if any(k in tl for k in ["lidar", "scan", "laserscan"]):
                        lidar_candidates.append(topic)

            if lidar_candidates and not self.has_lidar:
                self.lidar_topic = min(lidar_candidates, key=len)
                self.create_subscription(LaserScan, self.lidar_topic, self._lidar_cb, qos_profile_sensor_data)
                self.has_lidar = True
                self.get_logger().info(f"LIDAR: {self.lidar_topic}")

            # --- RGB CAMERA ---
            rgb_candidates = []
            for topic, types in topics.items():
                if "sensor_msgs/msg/Image" in types:
                    tl = topic.lower()
                    if any(k in tl for k in ["camera", "gimbal", "rgb", "image", "mono_cam"]):
                        if not any(k in tl for k in ["depth", "distance"]):
                            rgb_candidates.append(topic)

            if rgb_candidates and not self.has_rgb:
                self.rgb_topic = min(rgb_candidates, key=len)
                self.create_subscription(Image, self.rgb_topic, self._rgb_cb, qos_profile_sensor_data)
                self.has_rgb = True
                self.get_logger().info(f"RGB camera: {self.rgb_topic}")

            if self.has_depth or self.has_lidar or self.has_rgb:
                return

            time.sleep(1.0)
            rclpy.spin_once(self, timeout_sec=0.1)

        self.get_logger().error("No sensors found! Check ros_gz_bridge.")

    def _depth_cb(self, msg: Image):
        try:
            if msg.encoding == "32FC1":
                depth = np.frombuffer(msg.data, dtype=np.float32).reshape((msg.height, msg.width))
            elif msg.encoding == "16UC1":
                depth = np.frombuffer(msg.data, dtype=np.uint16).reshape((msg.height, msg.width)) / 1000.0
            else:
                return
        except Exception as e:
            self.get_logger().error(f"Depth callback error: {e}")
            return

        self.latest_depth = depth
        self.depth_count += 1
        self.frame_count += 1

        if self.airborne:
            self._process_depth(depth)

    def _process_depth(self, depth: np.ndarray):
        h, w = depth.shape
        now = time.time()

        if w != self.DEPTH_WIDTH or h != self.DEPTH_HEIGHT:
            depth = cv2.resize(depth, (self.DEPTH_WIDTH, self.DEPTH_HEIGHT))
            h, w = self.DEPTH_HEIGHT, self.DEPTH_WIDTH

        top = int(h * self.DEPTH_TOP_CROP)
        bottom = int(h * (1.0 - self.DEPTH_BOTTOM_CROP))
        left = int(w * self.DEPTH_EDGE_CROP)
        right = int(w * (1.0 - self.DEPTH_EDGE_CROP))

        if top >= bottom or left >= right:
            return

        depth_cropped = depth[top:bottom, left:right]
        h_c, w_c = depth_cropped.shape

        valid = (depth_cropped > self.DEPTH_MIN_M) & (depth_cropped < self.DEPTH_MAX_M) & ~np.isinf(depth_cropped) & ~np.isnan(depth_cropped)
        if not np.any(valid):
            return

        left_bound = int(w_c * self.REGION_LEFT)
        right_bound = int(w_c * self.REGION_RIGHT)

        regions = {
            "front_left": (depth_cropped[:, :left_bound], valid[:, :left_bound]),
            "front_center": (depth_cropped[:, left_bound:right_bound], valid[:, left_bound:right_bound]),
            "front_right": (depth_cropped[:, right_bound:], valid[:, right_bound:])
        }

        for name, (reg, reg_valid) in regions.items():
            if not np.any(reg_valid):
                continue

            valid_depths = reg[reg_valid]
            if len(valid_depths) < 10:
                continue

            dist = float(np.percentile(valid_depths, 10))

            if name in self.obstacle_ema:
                self.obstacle_ema[name] = (self.DISTANCE_EMA_ALPHA * dist + 
                                           (1 - self.DISTANCE_EMA_ALPHA) * self.obstacle_ema[name])
            else:
                self.obstacle_ema[name] = dist

            self.obstacles[name] = Obstacle(name, self.obstacle_ema[name], "depth", now)

    def _lidar_cb(self, msg: LaserScan):
        self.latest_scan = msg
        self.lidar_count += 1
        self.last_lidar_time = time.time()

        # Debug raw LIDAR stats every 25 scans
        if self.lidar_count % 25 == 0:
            ranges_arr = np.array(msg.ranges)
            dbg_valid = ~np.isinf(ranges_arr) & ~np.isnan(ranges_arr) & (ranges_arr > msg.range_min) & (ranges_arr < msg.range_max)
            if np.any(dbg_valid):
                self.get_logger().info(
                    f"LIDAR raw #{self.lidar_count}: beams={len(ranges_arr)}, valid={np.sum(dbg_valid)}, "
                    f"min={np.min(ranges_arr[dbg_valid]):.2f}, max={np.max(ranges_arr[dbg_valid]):.2f}, "
                    f"angle_inc={msg.angle_increment:.4f}"
                )
            else:
                self.get_logger().warn(
                    f"LIDAR raw #{self.lidar_count}: beams={len(ranges_arr)}, valid=0 — "
                    f"check if obstacles have collision geometry!"
                )

        try:
            now = time.time()
            ranges = np.array(msg.ranges)
            n = len(ranges)
            if n == 0 or msg.angle_increment == 0:
                return

            # Guaranteed exact length match, normalize to [0, 2π)
            angles = msg.angle_min + np.arange(n) * msg.angle_increment
            angles = np.mod(angles, 2 * np.pi)

            valid = ((ranges > max(msg.range_min, self.LIDAR_MIN_M)) & 
                     (ranges < min(msg.range_max, self.LIDAR_MAX_M)) & 
                     ~np.isinf(ranges) & ~np.isnan(ranges))

            for name, (deg_min, deg_max) in self.SECTORS.items():
                rad_min = np.mod(math.radians(deg_min), 2 * np.pi)
                rad_max = np.mod(math.radians(deg_max), 2 * np.pi)

                if rad_min > rad_max:
                    mask = ((angles >= rad_min) | (angles < rad_max)) & valid
                else:
                    mask = (angles >= rad_min) & (angles < rad_max) & valid

                if not np.any(mask):
                    continue

                sector_ranges = ranges[mask]
                dist = float(np.percentile(sector_ranges, 10))

                if name in self.obstacle_ema:
                    self.obstacle_ema[name] = (self.DISTANCE_EMA_ALPHA * dist + 
                                               (1 - self.DISTANCE_EMA_ALPHA) * self.obstacle_ema[name])
                else:
                    self.obstacle_ema[name] = dist

                if name not in self.obstacles or self.obstacle_ema[name] < self.obstacles[name].distance:
                    self.obstacles[name] = Obstacle(name, self.obstacle_ema[name], "lidar", now)

        except Exception as e:
            self.get_logger().error(f"LIDAR callback error: {e}")

    def _rgb_cb(self, msg: Image):
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
        except Exception as e:
            self.get_logger().error(f"RGB callback error: {e}")
            return

        self.latest_rgb = frame
        self.frame_count += 1

        if self.airborne and not self.has_depth:
            self._process_rgb_flow(frame)

    def _process_rgb_flow(self, frame: np.ndarray):
        small = cv2.resize(frame, (240, 160))
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        now = time.time()

        if self.prev_gray is None or self.prev_gray.shape != gray.shape:
            self.prev_gray = gray
            return

        mask = np.zeros_like(gray)
        mask[int(h*0.15):int(h*0.65), int(w*0.10):int(w*0.90)] = 255

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

        if mm < 0.3:
            return

        valid = mags < mm * 4
        p0 = p0[valid]
        flow = flow[valid]
        mags = mags[valid]

        x = p0[:, 0]
        cx = w / 2

        for name, xmask in [("front_left", x < cx * 0.7),
                            ("front_center", (x >= cx * 0.7) & (x < cx * 1.3)),
                            ("front_right", x >= cx * 1.3)]:
            if np.any(xmask):
                fm = mags[xmask]
                if len(fm) >= 3 and np.mean(fm) > 0.5:
                    dist = max(0.5, 10.0 / np.mean(fm))
                    if name in self.obstacle_ema:
                        self.obstacle_ema[name] = 0.3 * dist + 0.7 * self.obstacle_ema[name]
                    else:
                        self.obstacle_ema[name] = dist
                    self.obstacles[name] = Obstacle(name, self.obstacle_ema[name], "rgb", now)

    def _get_front_obstacles(self) -> List[Obstacle]:
        now = time.time()
        front_dirs = {"front_left", "front_center", "front_right"}
        obs = []
        for k, o in list(self.obstacles.items()):
            if now - o.timestamp > self.OBSTACLE_MEMORY_S:
                del self.obstacles[k]
                if k in self.obstacle_ema:
                    del self.obstacle_ema[k]
                continue
            if o.direction in front_dirs:
                obs.append(o)
        return obs

    def _get_all_obstacles(self) -> List[Obstacle]:
        now = time.time()
        obs = []
        for k, o in list(self.obstacles.items()):
            if now - o.timestamp > self.OBSTACLE_MEMORY_S:
                del self.obstacles[k]
                if k in self.obstacle_ema:
                    del self.obstacle_ema[k]
                continue
            obs.append(o)
        return obs

    def _get_closest(self, obstacles: List[Obstacle]) -> Optional[Obstacle]:
        if not obstacles:
            return None
        return min(obstacles, key=lambda o: o.distance)

    def _ramp_value(self, current: float, target: float, max_delta: float) -> float:
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
        all_obs = self._get_all_obstacles()
        closest_front = self._get_closest(front_obs)
        closest_all = self._get_closest(all_obs)

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

        react_dist = self.current_speed * self.REACT_TIME_S
        stop_dist = max(1.0, react_dist * 0.5)
        warn_dist = max(2.5, react_dist * 1.0)

        threat = False
        threat_dist = 999.0
        threat_dir = "none"

        if closest_front and (now - self.last_avoid_time) > 1.0:
            threat_dist = closest_front.distance
            threat_dir = closest_front.direction
            if threat_dist < stop_dist:
                threat = True
            elif threat_dist < warn_dist and self.state == State.CRUISE:
                threat = True

        if not threat and closest_all and closest_all.sensor == "lidar":
            if closest_all.distance < 1.5 and closest_all.direction in {"left", "right"}:
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
            target_yaw_rate = 0.0

        if self.state == State.CRUISE:
            if emergency:
                self.state = State.AVOID
                self.state_start_time = now
                self.avoid_direction = 1.0 if hdg_err > 0 else -1.0
                self.get_logger().error(f"EMERGENCY: {threat_dir} @ {closest_front.distance:.1f}m — backing up!")
            elif threat:
                self.state = State.AVOID
                self.state_start_time = now

                if "left" in threat_dir:
                    self.avoid_direction = 1.0
                elif "right" in threat_dir:
                    self.avoid_direction = -1.0
                else:
                    self.avoid_direction = 1.0 if hdg_err > 0 else -1.0

                self.get_logger().warn(
                    f"AVOID: {threat_dir} @ {threat_dist:.1f}m "
                    f"(speed={self.current_speed:.1f}, stop={stop_dist:.1f})"
                )
            else:
                target_speed = self.MAX_SPEED
                if closest_front:
                    d = closest_front.distance
                    if d < warn_dist:
                        target_speed = self.MIN_SPEED + (self.MAX_SPEED - self.MIN_SPEED) * (d / warn_dist)

                target_yaw_rate = float(np.clip(hdg_err * 1.2, -self.CRUISE_YAW_RATE, self.CRUISE_YAW_RATE))

        elif self.state == State.AVOID:
            if not threat or (closest_front and closest_front.distance > warn_dist * 1.2):
                self.state = State.CRUISE
                self.get_logger().info("→ CRUISE (clear)")
            elif now - self.state_start_time > 0.8:
                self.state = State.BYPASS
                self.state_start_time = now
                self.get_logger().info("→ BYPASS")
            else:
                target_speed = self.AVOID_FORWARD_SPEED if not emergency else -0.3
                target_lateral = self.avoid_direction * self.AVOID_LATERAL_SPEED * 0.5
                target_yaw_rate = self.avoid_direction * 20.0

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
                    self.get_logger().info("→ RECOVER")

            progress = elapsed / self.BYPASS_DURATION_S
            target_speed = self.BYPASS_FORWARD + (self.MAX_SPEED * 0.5 - self.BYPASS_FORWARD) * progress
            target_lateral = self.avoid_direction * self.AVOID_LATERAL_SPEED * (1.0 - progress * 0.3)
            yaw_toward_track = float(np.clip(hdg_err * 0.8, -10.0, 10.0))
            target_yaw_rate = self.avoid_direction * 5.0 + yaw_toward_track

        elif self.state == State.RECOVER:
            elapsed = now - self.state_start_time

            if elapsed > self.RECOVER_DURATION_S:
                self.state = State.CRUISE
                self.get_logger().info("→ CRUISE")

            target_speed = self.MAX_SPEED * 0.6
            target_yaw_rate = float(np.clip(hdg_err * 1.5, -self.CRUISE_YAW_RATE, self.CRUISE_YAW_RATE))

        self.current_speed = self._ramp_value(self.current_speed, target_speed, self.SPEED_RAMP_RATE)
        self.current_lateral = self._ramp_value(self.current_lateral, target_lateral, self.SPEED_RAMP_RATE * 0.8)
        self.current_yaw_rate = self._ramp_value(self.current_yaw_rate, target_yaw_rate, self.YAW_RAMP_RATE)

        return self.current_speed, self.current_lateral, 0.0, self.current_yaw_rate

    def _advance_wp(self):
        self.current_wp_idx = (self.current_wp_idx + 1) % len(self.WAYPOINTS)
        self.get_logger().info(f"WP → #{self.current_wp_idx} {self.WAYPOINTS[self.current_wp_idx]}")

    def get_vis(self):
        if self.latest_rgb is not None:
            vis = self.latest_rgb.copy()
        elif self.latest_depth is not None:
            d = self.latest_depth.copy()
            dvis = np.clip((np.nan_to_num(d, nan=0, posinf=0, neginf=0) / 10.0) * 255, 0, 255).astype(np.uint8)
            dvis = cv2.applyColorMap(dvis, cv2.COLORMAP_JET)
            vis = cv2.resize(dvis, (640, 480))
        else:
            return None

        h, w = vis.shape[:2]
        cv2.rectangle(vis, (0, 0), (w, 160), (0, 0, 0), -1)

        colors = {State.CRUISE: (0, 255, 0), State.SLOW: (0, 255, 255),
                  State.AVOID: (0, 165, 255), State.BYPASS: (0, 0, 255),
                  State.RECOVER: (255, 0, 255)}
        c = colors.get(self.state, (128, 128, 128))

        wp = self.WAYPOINTS[self.current_wp_idx]
        cv2.putText(vis, f"V12 {self.state.name} → WP{self.current_wp_idx}({wp[0]:.0f},{wp[1]:.0f})",
                   (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, c, 2)

        sensors = []
        if self.has_depth: sensors.append("DEPTH")
        if self.has_lidar: sensors.append("LIDAR")
        if self.has_rgb: sensors.append("RGB")
        cv2.putText(vis, f"Sensors: {'+'.join(sensors)}",
                   (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        closest = self._get_closest(self._get_all_obstacles())
        if closest:
            cv2.putText(vis, f"CLOSEST: {closest.direction} {closest.distance:.1f}m ({closest.sensor})",
                       (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        y_off = 90
        for obs in sorted(self._get_all_obstacles(), key=lambda o: o.distance)[:4]:
            cv2.putText(vis, f"  {obs.direction}: {obs.distance:.1f}m ({obs.sensor})",
                       (10, y_off), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)
            y_off += 14

        cv2.putText(vis, f"Speed: {self.current_speed:.1f} | YawRate: {self.current_yaw_rate:.1f} | Lat: {self.current_lateral:.1f}",
                   (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)

        if self.latest_depth is not None:
            overlay = vis.copy()
            b = int(h * self.DEPTH_BOTTOM_CROP)
            t = int(h * self.DEPTH_TOP_CROP)
            e = int(w * self.DEPTH_EDGE_CROP)
            cv2.rectangle(overlay, (0, h-b), (w, h), (0, 0, 255), -1)
            cv2.rectangle(overlay, (0, 0), (w, t), (0, 0, 255), -1)
            cv2.rectangle(overlay, (0, 0), (e, h), (0, 0, 255), -1)
            cv2.rectangle(overlay, (w-e, 0), (w, h), (0, 0, 255), -1)
            cv2.addWeighted(overlay, 0.2, vis, 0.8, 0, vis)

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
    node = DualSensorAvoidanceV12()
    threading.Thread(target=spin_ros2, args=(node,), daemon=True).start()

    drone = System()
    await drone.connect(system_address="udp://:14540")

    print("=" * 60)
    print("DUAL-SENSOR OBSTACLE AVOIDANCE v12.3")
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

    vis_path = os.path.expanduser("~/PX4-Autopilot/dual_sensor_v12.jpg")

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
            obs_str = ""
            if closest:
                obs_str = f" | {closest.direction}:{closest.distance:.1f}m"

            line = (f"\rState:{node.state.name:8s} | F:{fwd:5.2f} L:{lat:5.2f} | "
                   f"FL:{fl_s} FC:{fc_s} FR:{fr_s} | LIDAR:{lidar_health} | "
                   f"Pos:({pos_x:.1f},{pos_y:.1f}){obs_str}")
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
