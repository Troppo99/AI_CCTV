import cv2
from ultralytics import YOLO
import torch
from matplotlib import pyplot as plt

# Periksa apakah GPU tersedia dan set perangkat
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Memuat model YOLOv8
model = YOLO("../MY_FILES/Yolo-Models/yolov8l.pt")

# Memindahkan model ke perangkat yang sesuai
model.to(device)

# Jalur gambar lokal
img_path = "../MY_FILES/Datasets/Murtaza/Images/People.jpg"

# Melakukan prediksi
results = model.predict(source=img_path, device=device, conf=0.25)

# Menampilkan hasil prediksi menggunakan matplotlib
for result in results:
    # Asumsikan bahwa result.orig_img adalah gambar asli dengan bounding boxes
    img_with_boxes = (
        result.plot()
    )  # Metode 'plot' dapat digunakan untuk mendapatkan gambar dengan bounding boxes
    plt.imshow(cv2.cvtColor(img_with_boxes, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.show()
