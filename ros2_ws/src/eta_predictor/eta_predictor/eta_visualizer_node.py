#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
import time


class ETAVisualizer(Node):

    def __init__(self):
        super().__init__('eta_visualizer_node')

        self.csv_path = "/home/bangadmadhav/ros_projects/nav2-amr/ml/eta_results.csv"

        # Plot setup
        plt.show(block=False)
        self.fig, self.ax = plt.subplots(figsize=(10, 6))

        # Leaderboard print control
        self.last_print_time = 0

        # Timer
        self.create_timer(2.0, self.update_plot)

    # =========================================================
    # MAIN LOOP
    # =========================================================
    def update_plot(self):
        try:
            df = pd.read_csv(self.csv_path)
        except:
            return

        if len(df) < 2:
            return

        self.ax.clear()

        x = range(len(df))

        pred = df["predicted_time"]
        act = df["actual_time"]
        err = df["error_percent"]

        avg = err.expanding().mean()

        # ---------------- PLOTS ----------------
        self.ax.plot(x, pred, label="Predicted Time")
        self.ax.plot(x, act, label="Actual Time")
        self.ax.plot(x, err, label="Error %")
        self.ax.plot(x, avg, linestyle='--', label="Avg Error %")

        self.ax.set_title("ETA Prediction Performance")
        self.ax.set_xlabel("Run Index")
        self.ax.set_ylabel("Time / Error")
        self.ax.grid(True, alpha=0.3)
        self.ax.legend()

        plt.draw()
        plt.pause(0.01)

        # ---------------- LEADERBOARD ----------------
        # Print only every 10 seconds (avoid spam)
        if time.time() - self.last_print_time > 10:
            self.print_leaderboard(df)
            self.last_print_time = time.time()

    # =========================================================
    # LEADERBOARD
    # =========================================================
    def print_leaderboard(self, df):

        df_group = df.groupby(["planner", "controller"])

        leaderboard = df_group["error_percent"].agg(
            avg_error="mean",
            std_error="std",
            runs="count"
        ).reset_index()

        # Filter weak configs
        leaderboard = leaderboard[leaderboard["runs"] >= 3]

        if len(leaderboard) == 0:
            return

        # Ranking score (accuracy + stability)
        leaderboard["score"] = leaderboard["avg_error"] + 0.5 * leaderboard["std_error"]

        leaderboard = leaderboard.sort_values("score")

        print("\n" + "=" * 50)
        print("🏆 ETA PREDICTION LEADERBOARD")
        print("=" * 50)

        for rank, (_, row) in enumerate(leaderboard.head(5).iterrows(), start=1):

            status = "🟢" if row["avg_error"] < 3 else "🟡" if row["avg_error"] < 7 else "🔴"

            print(
                f"{rank}. {row['planner']} + {row['controller']} {status}\n"
                f"   Avg Error : {row['avg_error']:.2f}%\n"
                f"   Stability : {row['std_error']:.2f}\n"
                f"   Runs      : {int(row['runs'])}\n"
            )


# =========================================================
# MAIN
# =========================================================
def main():
    rclpy.init()
    node = ETAVisualizer()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()