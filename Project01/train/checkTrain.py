from ultralytics import YOLO

# Jalur ke file konfigurasi dataset
data_path = "D:/NWR27/AI_CCTV/ConstructionSiteSafety/data.yaml"

# Inisialisasi model YOLOv8 dari konfigurasi yaml
model = YOLO("yolov8l.yaml")  # Menggunakan konfigurasi arsitektur model dari file YAML

# Melatih model
model.train(
    data=data_path, epochs=50, imgsz=640, project="runs/detect", name="train_exp1"
)

# Model yang dilatih akan disimpan secara otomatis di direktori yang sesuai
