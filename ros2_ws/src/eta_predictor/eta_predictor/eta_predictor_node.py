#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseWithCovarianceStamped
from std_msgs.msg import Float32
from action_msgs.msg import GoalStatusArray

import numpy as np
import pandas as pd
import math
import joblib
import os
import time
from scipy.ndimage import distance_transform_edt
from PIL import Image
import yaml


class ETAPredictor(Node):

    def __init__(self):
        super().__init__('eta_predictor_node')

        # ---------------- FLAGS ----------------
        self.pose_ready = False
        self.pending_plan = None

        # ---------------- PATHS ----------------
        base_path = "/home/bangadmadhav/ros_projects/nav2-amr/ml/models"
        self.model = joblib.load(os.path.join(base_path, "gradient_boost_model.pkl"))
        self.feature_names = joblib.load(os.path.join(base_path, "feature_names.pkl"))

        self.get_logger().info("Gradient Boosting Model Loaded")

        # ---------------- PARAMETERS ----------------
        self.world = self.declare_parameter('world', 'world1').value
        self.planner = self.declare_parameter('planner', 'navfn').value
        self.controller = self.declare_parameter('controller', 'rpp').value
        self.max_vel = 0.5

        # ---------------- STATE ----------------
        self.current_pose = None

        # ---------------- MAP ----------------
        self.load_map()

        # ---------------- SUBSCRIBERS ----------------
        self.create_subscription(Path, '/plan', self.plan_callback, 10)
        self.create_subscription(PoseWithCovarianceStamped, '/amcl_pose', self.pose_callback, 10)

        # ---------------- PUBLISHERS ----------------
        self.pred_pub = self.create_publisher(Float32, '/predicted_time', 10)

    # =========================================================
    # MAP
    # =========================================================
    def load_map(self):
        map_yaml = f"/home/bangadmadhav/ros_projects/nav2-amr/ros2_ws/src/maps/{self.world}/map.yaml"

        with open(map_yaml, 'r') as f:
            map_data = yaml.safe_load(f)

        image_path = map_data['image']
        resolution = map_data['resolution']
        origin = map_data['origin']

        if not os.path.isabs(image_path):
            image_path = os.path.join(os.path.dirname(map_yaml), image_path)

        img = Image.open(image_path).convert('L')
        map_array = np.array(img)

        obstacle = map_array < 250
        free = 1 - obstacle

        self.distance_map = distance_transform_edt(free) * resolution
        self.map_resolution = resolution
        self.map_origin = origin
        self.map_height, self.map_width = map_array.shape

    def world_to_map(self, x, y):
        mx = int((x - self.map_origin[0]) / self.map_resolution)
        my = int((y - self.map_origin[1]) / self.map_resolution)

        my = self.map_height - my

        mx = max(0, min(mx, self.map_width - 1))
        my = max(0, min(my, self.map_height - 1))

        return my, mx

    # =========================================================
    # CALLBACKS
    # =========================================================
    def pose_callback(self, msg):
        self.current_pose = msg
        cov = msg.pose.covariance

        # ignore fake pose
        if (msg.pose.pose.position.x == 0.0 and
            msg.pose.pose.position.y == 0.0 and
            cov[0] == 0.0 and cov[7] == 0.0):
            return

        if cov[0] > 1e-4 or cov[7] > 1e-4:
            if not self.pose_ready:
                self.get_logger().info("AMCL ready (predictor)")
            self.pose_ready = True

        # 🔥 process pending plan
        if self.pending_plan is not None and self.pose_ready:
            self.process_plan(self.pending_plan)
            self.pending_plan = None

    def plan_callback(self, msg):

        if len(msg.poses) < 2:
            return

        # reset state for new goal
        self.prediction_active = False
        self.predicted_time = None

        if not self.pose_ready:
            self.pending_plan = msg
            return

        self.process_plan(msg)

    # =========================================================
    # CORE PIPELINE
    # =========================================================
    def process_plan(self, msg):

        if self.prediction_active:
            return

        # ---------------- FEATURES ----------------
        path_length, smoothness, turn_count = self.compute_path_features(msg)
        avg_clear, min_clear = self.compute_clearance(msg)
        uncertainty_raw = self.compute_uncertainty()
        log_uncertainty = math.log(uncertainty_raw + 1e-6)
        clearance_ratio = min_clear / (avg_clear + 1e-6)
        curvature = smoothness * turn_count

        # Adding Clippings to handle
        log_uncertainty = max(-5, min(-1, log_uncertainty))
        clearance_ratio = min(clearance_ratio, 1.5)

        data = {
            "path_length": path_length,
            "smoothness": smoothness,
            "turn_count": turn_count,
            "avg_clearance": avg_clear,
            "min_clearance": min_clear,
            "clearance_ratio": clearance_ratio,
            "curvature": curvature,
            "log_uncertainty": log_uncertainty
        }

        for f in self.feature_names:
            if "planner_" in f:
                data[f] = 1 if f == f"planner_{self.planner}" else 0
            elif "controller_" in f:
                data[f] = 1 if f == f"controller_{self.controller}" else 0

        X = pd.DataFrame([[data.get(f, 0) for f in self.feature_names]],
                         columns=self.feature_names)

        pred_ineff = self.model.predict(X)[0]

        ideal_time = path_length / self.max_vel if self.max_vel > 0 else 0
        predicted_time = ideal_time * pred_ineff

        # ---------------- STORE ----------------
        self.prediction_active = True   # optional safeguard

        # ---------------- PUBLISH ----------------
        msg_out = Float32()
        msg_out.data = float(predicted_time)
        self.pred_pub.publish(msg_out)

        self.get_logger().info(
            f"ETA → {predicted_time:.2f}s | "
            f"len={path_length:.2f} | "
            f"clear={avg_clear:.2f}/{min_clear:.2f} | "
            f"unc={log_uncertainty:.3f}"
        )

    # =========================================================
    # FEATURES
    # =========================================================
    def compute_path_features(self, msg):
        poses = msg.poses
        length = 0
        smoothness = 0
        turns = 0

        for i in range(1, len(poses)):
            x1 = poses[i-1].pose.position.x
            y1 = poses[i-1].pose.position.y
            x2 = poses[i].pose.position.x
            y2 = poses[i].pose.position.y

            dx = x2 - x1
            dy = y2 - y1

            length += math.hypot(dx, dy)

            if i > 1:
                prev = poses[i-2].pose.position
                a1 = math.atan2(y1 - prev.y, x1 - prev.x)
                a2 = math.atan2(dy, dx)

                dtheta = abs(self.angle_diff(a2, a1))
                smoothness += dtheta

                if dtheta > 0.3:
                    turns += 1

        smoothness /= max(1, len(poses)-1)
        return length, smoothness, turns

    def compute_clearance(self, msg):
        clearances = []
        for p in msg.poses:
            i, j = self.world_to_map(p.pose.position.x, p.pose.position.y)
            clearances.append(self.distance_map[i][j])
        return float(np.mean(clearances)), float(np.min(clearances))

    def compute_uncertainty(self):
        cov = self.current_pose.pose.covariance
        return cov[0] + cov[7] + cov[35]

    def angle_diff(self, a, b):
        return math.atan2(math.sin(a-b), math.cos(a-b))


def main(args=None):
    rclpy.init(args=args)
    node = ETAPredictor()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()