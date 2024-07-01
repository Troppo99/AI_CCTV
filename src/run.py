from main import main

main(
    # emp_model_path=".runs/detect/.arc/two_women/weights/best.pt",
    emp_model_path=".runs/detect/emp-m/weights/best.pt",
    act_model_path=".runs/detect/fold-m/weights/best.pt",
    # emp_classes=["Siti Umi", "Nina"],
    emp_classes=["Barden", "Deti", "Dita", "Fifi", "Nani", "Nina", "Siti-Umi", "Siti-Hutizah", "Sri-Anjani", "Tia"],
    act_classes=["Working"],
    # video_path="D:/AI_CCTV/.runs/videos/00000000302000000.mp4",
    # video_path=0,
    # mask_path=".runs/images/mask7.png",
    anto_time=3,
    # saver=True,
    send=True,
    host="10.5.0.2",
    # host="localhost",
    user="root",
    password="robot123",
    database="report_ai_cctv",
    port=3306,
    interval=1,
    table_sql="cam01",
)
