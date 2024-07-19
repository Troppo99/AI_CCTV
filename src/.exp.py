import json
import os
import time
from datetime import datetime


class REPORT:
    def __init__(self, classes, anto_time=3, backup_file=".runs/data/test/backup_data.json", data_loaded=True):
        self.data_loaded = data_loaded
        self.backup_file = backup_file

        if data_loaded == True:
            self.data, self.last_backup_date = self.load_backup_data(backup_file)
        else:
            self.data = {}
            self.last_backup_date = datetime.now().strftime("%Y-%m-%d")
            print(f"Data starts from zero")

        self.classes = classes
        self.anto_time = anto_time
        self.anomaly_tracker = {emp: {"idle": 0, "offsite": 0} for emp in classes}
        self.last_sent_time = time.time()

    def update_data(self, emp, act, frame_duration):
        current_date = datetime.now().strftime("%Y-%m-%d")

        if current_date != self.last_backup_date:
            self.reset_data()
            self.last_backup_date = current_date

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

        if self.data_loaded == True:
            self.backup_data()

    def backup_data(self):
        backup_content = {"date": self.last_backup_date, "data": self.data}
        with open(self.backup_file, "w") as file:
            json.dump(backup_content, file)

    @staticmethod
    def load_backup_data(backup_file):
        if os.path.exists(backup_file):
            with open(backup_file, "r") as file:
                backup_content = json.load(file)
            print(f"Data loaded from {backup_file}")
            return backup_content["data"], backup_content["date"]
        else:
            return {}, datetime.now().strftime("%Y-%m-%d")

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
