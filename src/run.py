from main import main

main(
    emp_model_path=".runs/detect/.arc/two_women/weights/best.pt",
    act_model_path=".runs/detect/fold-m/weights/best.pt",
    emp_classes=["Siti Umi", "Nina"],
    act_classes=["Working"],
    # video_path="D:/AI_CCTV/.runs/videos/00000000302000000.mp4",
    video_path=0,
    # mask_path=".runs/images/mask7.png",
)
