from main import main

main(
    emp_model_path=".runs/detect/emp-m/weights/best.pt",
    act_model_path=".runs/detect/fold-m/weights/best.pt",
    emp_classes=["Barden", "Deti", "Dita", "Fifi", "Nani", "Nina", "Umi", "Hutizah", "Anjani", "Tia"],
    act_classes=["Working"],
    # video_path="D:/AI_CCTV/.runs/videos/00000000302000000.mp4",
    mask_path=".runs/images/mask8.png",
    anto_time=300,
    # saver=True,
    table_sql="cam01",
    interval_send=1,
    server="10.5.0.3",
)
