from main import main

main(
    mask_path=".runs/images/mask8.png",
    anto_time=300,
    table_sql="cam01",
    interval_send=10,
    server="Waskita",
    # server="Nana",
    # saver=True,
    # video_path="D:/AI_CCTV/.runs/videos/00000000302000000.mp4",
)
