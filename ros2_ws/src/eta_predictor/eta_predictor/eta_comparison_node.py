#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32

import os
import csv
import time


class ETAComparison(Node):

    def __init__(self):
        super().__init__('eta_comparison_node')

        self.world = self.declare_parameter('world', 'world1').value
        self.planner = self.declare_parameter('planner', 'navfn').value
        self.controller = self.declare_parameter('controller', 'rpp').value

        self.predicted = None
        self.actual = None

        self.run_count = 1

        self.avg_errors = []
        self.csv_path = "/home/bangadmadhav/ros_projects/nav2-amr/ml/predictions/eta_results.csv"

        self.create_subscription(Float32, '/predicted_time', self.pred_cb, 10)
        self.create_subscription(Float32, '/actual_time', self.actual_cb, 10)

    def pred_cb(self, msg):
        self.predicted = msg.data
        self.try_print()

    def actual_cb(self, msg):
        self.actual = msg.data
        self.try_print()

    def bar(self, val, max_val, width=30):
        ratio = val / max_val
        n = int(ratio * width)
        bar = ["░"] * width
        if n < width:
            bar[n] = "|"
        return "".join(bar[:n+1])

    def try_print(self):
        if self.predicted is None or self.actual is None:
            return

        error = abs(self.predicted - self.actual)
        percent = (error / self.actual) * 100 if self.actual > 0 else 0
        max_val = max(self.predicted, self.actual, 1e-6)

        self.avg_errors.append(percent)
        avg_error_compute = sum(self.avg_errors) / len(self.avg_errors)

        if percent < 3:
            status = "🟢 EXCELLENT"
        elif percent < 7:
            status = "🟡 GOOD"
        else:
            status = "🔴 NEEDS IMPROVEMENT"


        self.get_logger().info(
            f"\n=== ETA COMPARISON (Run: {self.run_count}) ===\n"
            f"Predicted Time: {self.bar(self.predicted, max_val)} {self.predicted:.2f}s\n"
            f"Actual Time:    {self.bar(self.actual, max_val)} {self.actual:.2f}s\n"
            f"\nΔ Difference: {error:.2f}s\n"
            f"Error: ({percent:.2f}%) {status}\n"
            f"Average Error: {avg_error_compute:.2f}%\n"
        )
        # self.get_logger().info(
        #     f"\n=== ETA COMPARISON ===\n"
        #     f"Predicted: {self.predicted:.2f}s\n"
        #     f"Actual:    {self.actual:.2f}s\n"
        #     f"Error:     {error:.2f}s ({percent:.2f}%)\n"
        #     f"Status:    {status}\n"
        # )

        self.save_row(self.predicted, self.actual, error, percent)
        self.run_count += 1

        # reset for next run
        self.predicted = None
        self.actual = None
    
    def save_row(self, predicted, actual, error, percent):
        write_header = not os.path.exists(self.csv_path)

        row = {
            "trial_id": int(time.time() * 1000),
            "timestamp": time.time(),
            "world": self.world,
            "planner": self.planner,
            "controller": self.controller,
            "predicted_time": predicted,
            "actual_time": actual,
            "error_sec": error,
            "error_percent": percent
        }

        with open(self.csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())
            if write_header:
                writer.writeheader()
            writer.writerow(row)


def main(args=None):
    rclpy.init(args=args)
    node = ETAComparison()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()