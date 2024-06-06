from bsml3_sql import main

video_path = "../MY_FILES/Videos/CCTV/source/10_ch04_20240425073845.mp4"
model_emp_path, model_act_path = ".runs/detect/.arc/employees-1/weights/best.pt", ".runs/detect/.arc/eactivity-1/weights/best.pt"
emp_conf_th, act_conf_th = (0.8, 0.25)
video_scale = 0.75

main(video_path, model_emp_path, model_act_path, emp_conf_th, act_conf_th, video_scale)
