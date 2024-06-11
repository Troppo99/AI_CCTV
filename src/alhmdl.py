from ultralytics.data.annotator import auto_annotate

auto_annotate(data="ultralytics/assets", det_model="yolov8n.pt", sam_model="mobile_sam.pt")
