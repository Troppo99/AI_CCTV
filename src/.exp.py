import os
import time
import json
from datetime import datetime


class REPORT:
    def __init__(self, classes, anto_time=3, backup_folder=".runs/data/other/"):
        self.classes = classes
        self.anto_time = anto_time
        self.backup_folder = backup_folder
        self.anomaly_tracker = {emp: {"idle": 0, "offsite": 0} for emp in classes}
        self.last_sent_time = time.time()
        self.current_date = self.get_current_date()
        self.data = self.load_backup_data()

    @staticmethod
    def get_current_date():
        return datetime.now().strftime("%Y_%m_%d")

    def update_data(self, emp, act, frame_duration):
        if self.get_current_date() != self.current_date:
            self.backup_current_data()
            self.current_date = self.get_current_date()
            self.data = {emp: {"folding": 0, "idle": 0, "offsite": 0} for emp in self.classes}
            self.anomaly_tracker = {emp: {"idle": 0, "offsite": 0} for emp in self.classes}

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

        self.backup_data()

    def backup_current_data(self):
        backup_file = os.path.join(self.backup_folder, f"{self.current_date}.json")
        with open(backup_file, "w") as file:
            json.dump(self.data, file)
        print(f"Data backed up for date {self.current_date}.")

    def backup_data(self):
        backup_file = os.path.join(self.backup_folder, f"{self.current_date}.json")
        with open(backup_file, "w") as file:
            json.dump(self.data, file)

    def load_backup_data(self):
        current_date_file = os.path.join(self.backup_folder, f"{self.current_date}.json")
        if os.path.exists(current_date_file):
            with open(current_date_file, "r") as file:
                data = json.load(file)
            print(f"Data loaded from {current_date_file}")
            return data
        else:
            previous_files = sorted([f for f in os.listdir(self.backup_folder) if f.endswith(".json")], reverse=True)
            if previous_files:
                with open(os.path.join(self.backup_folder, previous_files[0]), "r") as file:
                    data = json.load(file)
                print(f"Data loaded from {previous_files[0]}")
                return data
            else:
                print("No backup data found, starting fresh.")
                return {emp: {"folding": 0, "idle": 0, "offsite": 0} for emp in self.classes}


report = REPORT(["Nana", "Nurdin"])

report.update_data("Nana", "folding", 10)
print(report.current_date)
