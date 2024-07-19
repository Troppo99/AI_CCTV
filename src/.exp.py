import time
from datetime import datetime
import os
import json


class REPORT:
    def __init__(self, classes, anto_time=3, backup_folder=".runs/data/other/"):
        self.classes = classes
        self.anto_time = anto_time
        self.backup_folder = backup_folder
        self.anomaly_tracker = {emp: {"idle": 0, "offsite": 0} for emp in classes}
        self.last_sent_time = time.time()
        self.current_date = self.get_current_date()
        self.data = self.load_backup_data()

    def get_current_date(self):
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
        with open(os.path.join(self.backup_folder, "backup_data.json"), "w") as file:
            json.dump(self.data, file)

    def load_backup_data(self):
        current_date_file = os.path.join(self.backup_folder, f"{self.current_date}.json")
        if os.path.exists(current_date_file):
            with open(current_date_file, "r") as file:
                data = json.load(file)
            print(f"Data loaded from {current_date_file}")
            return data
        else:
            backup_file = os.path.join(self.backup_folder, "backup_data.json")
            if os.path.exists(backup_file):
                with open(backup_file, "r") as file:
                    data = json.load(file)
                print(f"Data loaded from {backup_file}")
                return data
            else:
                print("No backup data found, starting fresh.")
                return {emp: {"folding": 0, "idle": 0, "offsite": 0} for emp in self.classes}



report = REPORT(["Nana", "Nurdin"])

# report.update_data("Nana", "folding", 100)  # Ensure data is generated and backed up
# print(report.current_date)  # This should print 2024_07_19

report.get_current_date = lambda:"2024_07_20"  # Force the date change
report.update_data("Nurdin", "folding", 100)  # Trigger the date change and backup
print(report.current_date)  # This should print 2024_07_20
