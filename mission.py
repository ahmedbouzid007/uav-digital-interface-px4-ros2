import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import numpy as np
import cv2
import os
import asyncio
from mavsdk import System
from mavsdk.mission import (MissionItem, MissionPlan)
from threading import Thread

class CameraListener(Node):
    def __init__(self):
        super().__init__('camera_listener')
        self.latest_image = None
        self.frame_count = 0
        
        # Hardcoded camera topic for forest world + x500_mono_cam
        camera_topic = '/world/forest/model/x500_mono_cam_0/link/camera_link/sensor/camera/image'
        
        self.get_logger().info(f"Listening to: {camera_topic}")
        self.create_subscription(Image, camera_topic, self.image_callback, 10)

    def image_callback(self, msg):
        try:
            # Convert ROS Image to OpenCV format
            if msg.encoding == 'rgb8':
                img = np.frombuffer(msg.data, dtype=np.uint8).reshape((msg.height, msg.width, 3))
                self.latest_image = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            elif msg.encoding == 'bgr8':
                self.latest_image = np.frombuffer(msg.data, dtype=np.uint8).reshape((msg.height, msg.width, 3))
            elif msg.encoding == 'mono8':
                self.latest_image = np.frombuffer(msg.data, dtype=np.uint8).reshape((msg.height, msg.width))
            else:
                img = np.frombuffer(msg.data, dtype=np.uint8).reshape((msg.height, msg.width, 3))
                self.latest_image = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            
            self.frame_count += 1
            
        except Exception as e:
            self.get_logger().error(f"Camera error: {e}")

def spin_ros2(node):
    rclpy.spin(node)

async def run_mission():
    # Initialize ROS2
    rclpy.init()
    node = CameraListener()
    
    # Spin ROS2 in background thread
    ros_thread = Thread(target=spin_ros2, args=(node,), daemon=True)
    ros_thread.start()
    
    # MAVSDK connection
    drone = System()
    await drone.connect(system_address="udp://:14540")
    
    final_path = os.path.expanduser("~/PX4-Autopilot/inspection_shot.jpg")
    
    print("Waiting for GPS...")
    async for health in drone.telemetry.health():
        if health.is_global_position_ok:
            print("GPS ready!")
            break
    
    print("Getting start position...")
    async for position in drone.telemetry.position():
        start_lat, start_lon = position.latitude_deg, position.longitude_deg
        break 
    
    # Target: 5 meters North
    target_lat = start_lat + 0.000045 
    target_lon = start_lon

    def create_item(lat, lon, alt):
        return MissionItem(
            lat, lon, float(alt), 3.0, 
            False,  # stop at waypoint
            float('nan'), float('nan'),
            MissionItem.CameraAction.TAKE_PHOTO,
            5.0,  # loiter time for stabilization
            float('nan'), float('nan'), float('nan'), float('nan'),
            MissionItem.VehicleAction.NONE
        )

    p1 = create_item(start_lat, start_lon, 10.0)
    p2 = create_item(target_lat, target_lon, 10.0)

    print("-- Uploading mission...")
    await drone.mission.upload_mission(MissionPlan([p1, p2]))

    print("-- Arming and starting mission...")
    await drone.action.arm()
    await asyncio.sleep(1)
    await drone.mission.start_mission()

    async for progress in drone.mission.mission_progress():
        if progress.current == progress.total:
            print("-- Target reached. Stabilizing camera...")
            await asyncio.sleep(3)  # Wait for gimbal/drone to stabilize
            
            # Save ONLY ONE frame at final checkpoint
            if node.latest_image is not None:
                cv2.imwrite(final_path, node.latest_image)
                print(f"--- SUCCESS: Single image saved to {final_path} ---")
                print(f"Total frames received during flight: {node.frame_count}")
                print(f"Saved only the LAST frame at waypoint")
            else:
                print("--- ERROR: No camera image received ---")
            
            break

    print("-- Mission complete. Returning to launch...")
    await drone.action.return_to_launch()
    
    # Cleanup
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    asyncio.run(run_mission())
