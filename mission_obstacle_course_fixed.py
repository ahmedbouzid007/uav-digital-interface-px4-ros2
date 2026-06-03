import asyncio
import math
import os
import threading
import time

import cv2
import numpy as np
import rclpy
from mavsdk import System
from mavsdk.offboard import OffboardError, VelocityBodyYawspeed
from rclpy.node import Node
from sensor_msgs.msg import Image


class OpticalFlowAvoidanceFinal(Node):
    def __init__(self):
        super().__init__("optical_flow_avoidance_final")

        self.possible_topics = [
            "/world/obstacle_course/model/x500_gimbal_0/link/camera_link/sensor/camera/image",
            "/world/obstacle_course/model/x500_mono_cam_0/link/camera_link/sensor/camera/image",
            "/world/obstacle_course/model/x500_depth_0/link/camera_link/sensor/camera/image",
        ]
        self.camera_topic = None
        self.subscription = None

        self.latest_image = None
        self.frame_count = 0
        self.receiving_images = False
        self.last_image_time = 0

        self.prev_gray = None
        self.min_features = 6

        self.feature_params = dict(
            maxCorners=200, qualityLevel=0.3, minDistance=7, blockSize=7
        )
        self.lk_params = dict(
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
        )

        self.home_lat = None
        self.home_lon = None
        self.METER_TO_DEG = 1.0 / 111319.5

        self.airborne = False
        self.altitude = 10.0

        # Optical flow detection state
        self.obstacle_detected = False
        self.obstacle_direction = None
        self.obstacle_confidence = 0.0
        self.flow_lateral = 0.0

        self._subscribe_camera()

    def _subscribe_camera(self):
        if self.camera_topic is not None:
            return

        available_topics = [name for name, _ in self.get_topic_names_and_types()]
        for topic in self.possible_topics:
            self.get_logger().info(f"Trying camera topic: {topic}")
            if topic in available_topics:
                self.camera_topic = topic
                self.subscription = self.create_subscription(
                    Image, topic, self.image_callback, 10
                )
                self.get_logger().info(f"SUCCESS: Subscribed to {topic}")
                return
            self.get_logger().warn(f"Topic not available yet: {topic}")

        self.get_logger().error("No camera topic available!")
        self.get_logger().error("Start bridge with:")
        self.get_logger().error(
            "  ros2 run ros_gz_bridge parameter_bridge "
            "/world/obstacle_course/model/x500_gimbal_0/link/camera_link/sensor/camera/image"
            "@sensor_msgs/msg/Image[gz.msgs.Image"
        )

    def retry_camera_subscription(self, retries=15, delay=1.0):
        for _ in range(retries):
            if self.camera_topic is not None:
                return True
            self._subscribe_camera()
            if self.camera_topic is not None:
                return True
            time.sleep(delay)
            rclpy.spin_once(self, timeout_sec=0.1)
        return self.camera_topic is not None

    def set_home_position(self, lat, lon):
        self.home_lat = lat
        self.home_lon = lon
        self.get_logger().info(f"Home set: lat={lat:.8f}, lon={lon:.8f}")

    def set_airborne(self, airborne, altitude=10.0):
        self.airborne = airborne
        self.altitude = altitude

    def _gps_to_local(self, lat, lon):
        """Convert GPS to local ENU coordinates (East, North) in meters."""
        if self.home_lat is None or self.home_lon is None:
            return 0.0, 0.0
        dlat = lat - self.home_lat
        dlon = lon - self.home_lon
        north = dlat / self.METER_TO_DEG
        east = dlon / (self.METER_TO_DEG * math.cos(math.radians(self.home_lat)))
        return east, north

    def _local_to_gps(self, east, north):
        """Convert local ENU (East, North) back to GPS."""
        if self.home_lat is None or self.home_lon is None:
            return 0.0, 0.0
        dlat = north * self.METER_TO_DEG
        dlon = east * self.METER_TO_DEG * math.cos(math.radians(self.home_lat))
        return self.home_lat + dlat, self.home_lon + dlon

    def image_callback(self, msg):
        """Process camera images and run optical flow in the callback thread."""
        try:
            data = np.frombuffer(msg.data, dtype=np.uint8)
            if msg.encoding == "rgb8":
                frame = data.reshape((msg.height, msg.width, 3))[:, :, ::-1]
            elif msg.encoding == "bgr8":
                frame = data.reshape((msg.height, msg.width, 3))
            elif msg.encoding == "mono8":
                gray = data.reshape((msg.height, msg.width))
                frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            else:
                self.get_logger().error(f"Unsupported encoding: {msg.encoding}")
                return
        except Exception as exc:
            self.get_logger().error(f"Image conversion failed: {exc}")
            return

        self.latest_image = frame
        self.frame_count += 1
        self.receiving_images = True
        self.last_image_time = time.time()

        # Process optical flow asynchronously in the callback
        if self.airborne and self.altitude > 3.0:
            self._process_optical_flow(frame)

    def _process_optical_flow(self, frame):
        """Compute optical flow and detect obstacles from divergence."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self.prev_gray is None:
            self.prev_gray = gray
            self.obstacle_detected = False
            self.flow_lateral = 0.0
            return

        p0 = cv2.goodFeaturesToTrack(self.prev_gray, mask=None, **self.feature_params)
        self.prev_gray = gray

        if p0 is None or len(p0) < self.min_features:
            self.obstacle_detected = False
            self.flow_lateral = 0.0
            return

        p1, st, err = cv2.calcOpticalFlowPyrLK(
            self.prev_gray, gray, p0, None, **self.lk_params
        )
        if p1 is None or st is None:
            self.obstacle_detected = False
            self.flow_lateral = 0.0
            return

        p0 = p0.reshape(-1, 2)
        p1 = p1.reshape(-1, 2)
        st = st.reshape(-1)

        good_new = p1[st == 1]
        good_old = p0[st == 1]
        if len(good_new) < self.min_features:
            self.obstacle_detected = False
            self.flow_lateral = 0.0
            return

        flow_vectors = good_new - good_old
        magnitudes = np.linalg.norm(flow_vectors, axis=1)
        if len(magnitudes) == 0:
            self.obstacle_detected = False
            self.flow_lateral = 0.0
            return

        median_mag = np.median(magnitudes)
        if median_mag < 0.5:
            self.obstacle_detected = False
            self.flow_lateral = 0.0
            return

        # Filter outliers
        valid = magnitudes < median_mag * 3.0
        if np.count_nonzero(valid) < self.min_features:
            self.obstacle_detected = False
            self.flow_lateral = 0.0
            return

        good_new = good_new[valid]
        magnitudes = magnitudes[valid]

        # Divide into left/center/right regions
        w = gray.shape[1]
        x = good_new[:, 0]

        left_mask = x < w * 0.33
        right_mask = x >= w * 0.66
        center_mask = ~(left_mask | right_mask)

        left_score = np.median(magnitudes[left_mask]) if left_mask.any() else 0.0
        right_score = np.median(magnitudes[right_mask]) if right_mask.any() else 0.0
        center_score = np.median(magnitudes[center_mask]) if center_mask.any() else 0.0

        # Detect obstacle based on flow magnitude
        threshold = 2.0
        if center_score > max(left_score, right_score) * 1.2 and center_score > threshold:
            self.obstacle_detected = True
            self.obstacle_direction = "center"
            self.obstacle_confidence = min(center_score / 10.0, 1.0)
            self.flow_lateral = 2.0
        elif left_score > right_score * 1.25 and left_score > threshold:
            self.obstacle_detected = True
            self.obstacle_direction = "left"
            self.obstacle_confidence = min(left_score / 10.0, 1.0)
            self.flow_lateral = -1.5
        elif right_score > left_score * 1.25 and right_score > threshold:
            self.obstacle_detected = True
            self.obstacle_direction = "right"
            self.obstacle_confidence = min(right_score / 10.0, 1.0)
            self.flow_lateral = 1.5
        else:
            self.obstacle_detected = False
            self.obstacle_direction = None
            self.obstacle_confidence = 0.0
            self.flow_lateral = 0.0

        if self.obstacle_detected:
            self.get_logger().warning(
                f"FLOW: {self.obstacle_direction.upper()} | "
                f"L={left_score:.1f} C={center_score:.1f} R={right_score:.1f} | "
                f"lat_cmd={self.flow_lateral:.1f}"
            )

    def get_avoidance_velocity(self, forward_speed=2.0):
        """Get velocity with optical flow avoidance."""
        if not self.airborne:
            return forward_speed, 0.0, 0.0

        if self.obstacle_detected:
            safe_forward = forward_speed * max(0.0, 1.0 - self.obstacle_confidence * 0.8)
            lateral = self.flow_lateral * (1.0 + self.obstacle_confidence)
            self.get_logger().warning(
                f"AVOID: F={safe_forward:.1f} L={lateral:.1f} conf={self.obstacle_confidence:.1%}"
            )
            return safe_forward, lateral, 0.0

        return forward_speed, 0.0, 0.0

    def get_visualization(self):
        """Create debug visualization."""
        if self.latest_image is None:
            return None

        vis = self.latest_image.copy()

        if not self.airborne:
            color = (128, 128, 128)
            text = "GROUND"
        elif self.obstacle_detected:
            color = (0, 0, 255)
            text = f"AVOID: {self.obstacle_direction.upper()} ({self.obstacle_confidence:.0%})"
        else:
            color = (0, 255, 0)
            text = "CLEAR"

        cv2.rectangle(vis, (5, 5), (400, 80), (0, 0, 0), -1)
        cv2.putText(vis, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(vis, f"Frames: {self.frame_count}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        return vis


def spin_ros2(node):
    rclpy.spin(node)


def normalize_angle(angle):
    """Normalize angle to [0, 360) range."""
    return angle % 360


def heading_error(target, current):
    """Compute shortest heading difference with proper wrap-around.

    Returns error in [-180, 180] where positive means turn clockwise
    (increase heading) to reach target from current.
    """
    target = normalize_angle(target)
    current = normalize_angle(current)
    diff = target - current
    if diff > 180:
        diff -= 360
    elif diff < -180:
        diff += 360
    return diff


async def run_mission_final():
    rclpy.init()
    node = OpticalFlowAvoidanceFinal()

    ros_thread = threading.Thread(target=spin_ros2, args=(node,), daemon=True)
    ros_thread.start()

    drone = System()
    await drone.connect(system_address="udp://:14540")

    print("Waiting for GPS...")
    async for health in drone.telemetry.health():
        if health.is_global_position_ok:
            print("GPS ready!")
            break

    print("Getting start position...")
    start_lat = None
    start_lon = None
    async for position in drone.telemetry.position():
        if abs(position.latitude_deg) > 0.0 and abs(position.longitude_deg) > 0.0:
            start_lat = position.latitude_deg
            start_lon = position.longitude_deg
            print(f"Start: lat={start_lat:.8f}, lon={start_lon:.8f}")
            break

    if start_lat is None or start_lon is None:
        raise RuntimeError("Could not read start GPS position")

    node.set_home_position(start_lat, start_lon)

    if not node.camera_topic:
        print("Retrying camera subscription...")
        node.retry_camera_subscription()

    vis_path = os.path.expanduser("~/PX4-Autopilot/optical_flow_final_vis.jpg")

    # ============================================================
    # WAYPOINTS THAT GO AROUND OBSTACLES — NOT THROUGH THEM
    # 
    # Obstacle map (from SDF):
    #   wall_01:     (East=8,  North=0)   size 0.5×8×6  → occupies Y∈[-4,4] at X=8
    #   rock_01:     (East=5,  North=8)    radius ~0.6
    #   tree_01:     (East=15, North=10)   radius ~2.0 (trunk) + 2.0 (leaves)
    #   pillar_01:   (East=12, North=-8)   radius ~0.5
    #   rock_02:     (East=-8, North=-3)   radius ~0.6
    #   wall_02:     (East=-5, North=12)   size 0.5×6×5  → rotated 90°
    #   building_01: (East=-15,North=-5)   size 10×10×10
    #   building_02: (East=25, North=20)   size 8×12×8
    #   tree_02:     (East=-10,North=15)   radius ~2.5
    #   tree_03:     (East=20, North=-10)  radius ~1.8
    #   pillar_02:   (East=-12,North=-15)  radius ~0.6
    #   rock_03:     (East=18, North=5)    radius ~0.5
    #
    # Strategy: Fly a large square path that passes NEAR obstacles 
    # (to trigger optical flow) but with 5m+ clearance (to avoid collision)
    # ============================================================

    waypoints_local = [
        # Start
        (0, 0, "START"),

        # Go North first, well clear of wall_01 at (8,0)
        (10, 0, "North 10m — clear of wall_01"),

        # Pass East of rock_01 at (5,8), staying at East=10
        (10, 5, "East of rock_01"),
        (10, 10, "Continue East"),

        # Approach tree_01 at (15,10) from North-West, pass North of it
        (12, 12, "North-West of tree_01"),
        (15, 15, "North of tree_01"),
        (18, 15, "East of tree_01"),

        # Head South-East toward tree_03 at (20,-10), pass East of it
        (20, 10, "Transit SE"),
        (22, 5, "Approach tree_03 from North"),
        (22, -5, "South of tree_03"),

        # Head West, passing South of pillar_01 at (12,-8)
        (18, -8, "South-East of pillar_01"),
        (15, -10, "South of pillar_01"),
        (10, -10, "Continue West"),

        # Pass South of wall_01 at (8,0) — now going West at Y=-5
        (5, -5, "South of wall_01"),
        (0, -5, "West of wall_01"),

        # Head toward rock_02 at (-8,-3), pass South of it
        (-5, -5, "East of rock_02"),
        (-8, -5, "South of rock_02"),
        (-10, -5, "West of rock_02"),

        # Head North toward building_01 at (-15,-5), pass East of it
        (-12, 0, "South-East of building_01"),
        (-12, 5, "East of building_01"),

        # Head North-East toward tree_02 at (-10,15), pass South-East of it
        (-8, 10, "South of tree_02"),
        (-5, 12, "South-East of tree_02"),
        (-5, 15, "East of tree_02"),

        # Head East toward wall_02 at (-5,12), pass South of it
        (0, 12, "South of wall_02"),
        (5, 12, "East of wall_02"),

        # Head South-East toward building_02 at (25,20), pass North-West of it
        (10, 15, "Transit toward building_02"),
        (15, 18, "West of building_02"),
        (20, 20, "North-West of building_02"),

        # Return home via South
        (15, 15, "Transit SW"),
        (10, 10, "Continue SW"),
        (5, 5, "More SW"),
        (0, 0, "HOME"),
    ]

    # Convert local waypoints to GPS
    absolute_waypoints = []
    for north_m, east_m, desc in waypoints_local:
        lat, lon = node._local_to_gps(east_m, north_m)
        absolute_waypoints.append((lat, lon, desc, east_m, north_m))

    print(f"\n=== Mission with {len(absolute_waypoints)} waypoints ===")
    for idx, (_, _, desc, east, north) in enumerate(absolute_waypoints):
        print(f"  WP{idx}: {desc} (East={east}m, North={north}m)")

    print("\n-- Setting initial setpoint...")
    await drone.offboard.set_velocity_body(
        VelocityBodyYawspeed(0.0, 0.0, 0.0, 0.0)
    )

    print("-- Starting offboard mode...")
    try:
        await drone.offboard.start()
    except OffboardError as error:
        print(f"Failed to start offboard: {error._result.result}")
        return

    print("-- Arming...")
    await drone.action.arm()
    await asyncio.sleep(2)

    print("-- Taking off to 10m...")
    await drone.offboard.set_velocity_body(
        VelocityBodyYawspeed(0.0, 0.0, -2.0, 0.0)
    )
    await asyncio.sleep(5)
    await drone.offboard.set_velocity_body(
        VelocityBodyYawspeed(0.0, 0.0, 0.0, 0.0)
    )
    await asyncio.sleep(2)

    print("\n-- Waiting for camera images (up to 60s)...")
    camera_ready = False
    for _ in range(60):
        if node.frame_count > 5:
            camera_ready = True
            break
        await asyncio.sleep(1.0)

    if not camera_ready:
        print("WARNING: Camera not receiving! Check ros_gz_bridge.")
    else:
        print(f"Camera OK! {node.frame_count} frames")

    node.set_airborne(True, altitude=10.0)

    # NAVIGATION LOOP
    current_wp_idx = 1
    prev_dist = float('inf')
    stuck_counter = 0

    while current_wp_idx < len(absolute_waypoints):
        target_lat, target_lon, desc, target_east, target_north = absolute_waypoints[current_wp_idx]
        print(f"\n-- Navigating to WP{current_wp_idx}: {desc} (E={target_east}, N={target_north})")

        wp_start_time = time.time()
        max_wp_time = 60  # 60 seconds per waypoint
        prev_dist = float('inf')
        stuck_counter = 0

        while time.time() - wp_start_time < max_wp_time:
            # Get current position
            current_lat = None
            current_lon = None
            async for position in drone.telemetry.position():
                current_lat = position.latitude_deg
                current_lon = position.longitude_deg
                break

            # Get current heading
            current_heading = None
            async for hdg in drone.telemetry.heading():
                current_heading = hdg.heading_deg
                break

            if current_lat is None or current_lon is None or current_heading is None:
                await asyncio.sleep(0.1)
                continue

            # Compute distance to target in local coordinates
            current_east, current_north = node._gps_to_local(current_lat, current_lon)
            dist = math.hypot(target_east - current_east, target_north - current_north)

            # Detect if we're moving away from waypoint (stuck/spinning)
            if dist > prev_dist + 0.5:
                stuck_counter += 1
                if stuck_counter > 10:
                    print(f"  WARNING: Moving away from WP{current_wp_idx} (dist={dist:.1f}m > prev={prev_dist:.1f}m). Reducing yaw authority.")
            else:
                stuck_counter = max(0, stuck_counter - 1)
            prev_dist = dist

            # Compute heading to target
            d_north = target_north - current_north
            d_east = target_east - current_east
            target_heading = math.degrees(math.atan2(d_east, d_north))

            # FIXED: Proper heading error with wrap-around
            h_err = heading_error(target_heading, current_heading)

            # Speed scaling
            if dist < 2.0:
                forward_speed = 0.5
            elif dist < 5.0:
                forward_speed = 1.0
            else:
                forward_speed = 2.0

            # Get avoidance velocities from optical flow
            avoid_forward, avoid_lateral, _ = node.get_avoidance_velocity(forward_speed=forward_speed)

            # FIXED: Smooth yaw control with anti-spin protection
            # If stuck (moving away), reduce yaw rate to prevent spin-out
            yaw_gain = 0.8 if stuck_counter > 5 else 1.0
            max_yaw_rate = 30.0 if stuck_counter > 5 else 45.0
            yaw_rate = np.clip(h_err * yaw_gain, -max_yaw_rate, max_yaw_rate)

            # Body frame velocity command
            # If heading error is large, turn in place with minimal forward motion
            if abs(h_err) > 90:
                # Large error: stop forward motion, turn aggressively but not max
                forward_cmd = 0.0
                lateral_cmd = 0.0
                yaw_cmd = yaw_rate
            elif abs(h_err) > 45:
                # Medium error: minimal forward, focus on turning
                forward_cmd = 0.2
                lateral_cmd = 0.0
                yaw_cmd = yaw_rate
            elif abs(h_err) > 20:
                # Small-medium error: half forward, turn
                forward_cmd = avoid_forward * 0.5
                lateral_cmd = avoid_lateral
                yaw_cmd = yaw_rate
            else:
                # Small error: full forward, gentle turn
                forward_cmd = avoid_forward
                lateral_cmd = avoid_lateral
                yaw_cmd = yaw_rate * 0.5

            await drone.offboard.set_velocity_body(
                VelocityBodyYawspeed(forward_cmd, lateral_cmd, 0.0, yaw_cmd)
            )

            # Status print
            status = (
                f"  WP{current_wp_idx} | Dist: {dist:.1f}m | "
                f"Heading: {current_heading:.0f}° → {normalize_angle(target_heading):.0f}° (err={h_err:.0f}°) | "
                f"Vel: F={forward_cmd:.1f} L={lateral_cmd:.1f} Y={yaw_cmd:.1f} | "
            )
            if node.obstacle_detected:
                status += f"AVOID: {node.obstacle_direction} ({node.obstacle_confidence:.0%})"
            else:
                status += "CLEAR"
            print(f"\r{status}", end="", flush=True)

            # Save visualization periodically
            if node.frame_count % 30 == 0 and node.latest_image is not None:
                vis = node.get_visualization()
                if vis is not None:
                    cv2.imwrite(vis_path, vis)

            # Check if reached
            if dist < 1.5:
                print(f"\n  --> WP{current_wp_idx} REACHED! Dist: {dist:.1f}m")
                current_wp_idx += 1
                break

            await asyncio.sleep(0.1)

        else:
            # Timeout reached
            print(f"\n  --> WP{current_wp_idx} TIMEOUT after {max_wp_time}s")
            current_wp_idx += 1

    # MISSION COMPLETE
    print("\n-- Mission complete!")
    node.set_airborne(False)
    await drone.offboard.set_velocity_body(
        VelocityBodyYawspeed(0.0, 0.0, 0.0, 0.0)
    )
    await asyncio.sleep(3)

    if node.latest_image is not None:
        vis = node.get_visualization()
        if vis is not None:
            cv2.imwrite(vis_path, vis)
        print(f"Saved visualization to {vis_path}")

    print("\n-- Landing...")
    await drone.offboard.set_velocity_body(
        VelocityBodyYawspeed(0.0, 0.0, 1.0, 0.0)
    )
    await asyncio.sleep(5)

    try:
        await drone.offboard.stop()
    except OffboardError as error:
        print(f"Failed to stop offboard: {error._result.result}")

    await drone.action.disarm()
    node.destroy_node()
    rclpy.shutdown()
    print("\n=== Mission Complete ===")


if __name__ == "__main__":
    asyncio.run(run_mission_final())

