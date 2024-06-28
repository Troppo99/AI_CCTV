import bsml4

bsml4.main(
    mask_path=".runs/images/mask7.png",
    emp_model_path=".runs/detect/.arc/two_women/weights/best.pt",
    act_model_path=".runs/weights/fold.pt",
    emp_classes=["Siti Umi", "Nina"],
    act_classes=["Folding", "Idle"],
    video_path="D:/AI_CCTV/.runs/videos/00000000302000000.mp4",
)
