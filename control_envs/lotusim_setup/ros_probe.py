"""Discovery probe: start an rclpy node, wait a few seconds and print the
topics, services and action servers it can see. Run inside the dev shell:

    nix develop path:control_envs/lotusim_setup --command python3 control_envs/lotusim_setup/ros_probe.py
"""
import os
import time

import rclpy

rclpy.init()
node = rclpy.create_node("probe")
print("rmw:", rclpy.utilities.get_rmw_implementation_identifier())
for k in ("ROS_DOMAIN_ID", "ROS_LOCALHOST_ONLY", "ROS_AUTOMATIC_DISCOVERY_RANGE",
          "FASTDDS_BUILTIN_TRANSPORTS", "RMW_IMPLEMENTATION"):
    print(f"  {k}={os.environ.get(k)}")
t0 = time.time()
while time.time() - t0 < 8.0:
    rclpy.spin_once(node, timeout_sec=0.5)
topics = node.get_topic_names_and_types()
print(f"topics ({len(topics)}):")
for n, t in topics:
    print("  ", n, t)
print("nodes:", node.get_node_names_and_namespaces())
srv = node.get_service_names_and_types()
print(f"services ({len(srv)}):", [n for n, _ in srv if "mas" in n or "lotus" in n][:10])
node.destroy_node()
rclpy.shutdown()
