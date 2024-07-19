import json
import os
import time
from datetime import datetime


class REPORT:
    def __init__(self, classes, anto_time=3, backup_dir=".runs/data/test/", data_loaded=True):
        self.data_loaded = data_loaded
        self.backup_dir = backup_dir
        self.current_date = datetime.now().strftime("%Y_%m_%d")
        self.backup_file = os.path.join(backup_dir, f"{self.current_date}.json")

        if data_loaded and os.path.exists(self.backup_file):
            self.data = self.load_backup_data(self.backup_file)
        else:
            self.data = {}
            print("Data starts from zero")

        self.classes = classes
        self.anto_time = anto_time
        self.anomaly_tracker = {emp: {"idle": 0, "offsite": 0} for emp in classes}
        self.last_sent_time = time.time()

    def update_data(self, emp, act, frame_duration):
        current_date = datetime.now().strftime("%Y_%m_%d")

        if current_date != self.current_date:
            self.current_date = current_date
            self.backup_file = os.path.join(self.backup_dir, f"{self.current_date}.json")
            self.reset_data()

        if emp not in self.data:
            self.data[emp] = {
                "folding": 0,
                "idle": 0,
                "offsite": 0,
            }

        if act == "folding":
            self.data[emp]["folding"] += frame_duration
            self.anomaly_tracker[emp]["idle"] = 0
            self.anomaly_tracker[emp]["offsite"] = 0
        elif act == "idle":
            self.anomaly_tracker[emp]["idle"] += frame_duration
            if self.anomaly_tracker[emp]["idle"] > self.anto_time:
                self.data[emp]["idle"] += frame_duration
        elif act == "offsite":
            self.anomaly_tracker[emp]["offsite"] += frame_duration
            if self.anomaly_tracker[emp]["offsite"] > self.anto_time:
                self.data[emp]["offsite"] += frame_duration

        if self.data_loaded:
            self.backup_data()

    def backup_data(self):
        with open(self.backup_file, "w") as file:
            json.dump(self.data, file)

    @staticmethod
    def load_backup_data(backup_file):
        with open(backup_file, "r") as file:
            data = json.load(file)
        print(f"Data loaded from {backup_file}")
        return data

    def reset_data(self):
        self.data = {emp: {"folding": 0, "idle": 0, "offsite": 0} for emp in self.classes}
        self.anomaly_tracker = {emp: {"idle": 0, "offsite": 0} for emp in self.classes}
        self.backup_data()
        print("Data has been reset due to day change")


# Example usage:
report = REPORT(classes=["Alice", "Bob"], data_loaded=True)
report.update_data("Alice", "folding", 5)
report.update_data("Alice", "idle", 4)
report.update_data("Bob", "offsite", 6)
