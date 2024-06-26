""" * * *---> Start of experiment 1 <---* * * """

# my_dict = {"Nana": {"folding": 1, "idle": 0}, "Wartana": {"folding": 5, "idle": 3}}

# # print(my_dict["Nana"].values())
# for emp_class in my_dict["Nana"].keys():
#   print(emp_class)
""" --------> End of experiment 1 <-------- """


""" * * *---> Start of experiment 2 <---* * * """
# Dictionary dengan beberapa kunci
# emp_classes = ["Nana", "Was"]
# data = {
#     "Nana": {"Folding": 5, "Idle": 0},
#     "Was": {"Folding": 0, "Idle": 5},
# }
# for emp_class in data:
#     for key in data[emp_class]:
#         percentage_key = f"%{key[0]}"
#         print(percentage_key)
#     break
""" --------> End of experiment 2 <-------- """

""" * * *---> Start of experiment 3 <---* * * """


# class Negara:
#     def __init__(self, city="Majalengka"):
#         self.kota = city
#         # self.desaKu() # ini ada hubungannya dengan ^^^^

#     def desaKu(self, desa="Mekarmulya", dusun="Cisahang"):
#         self.desa = desa
#         self.dusun = dusun


# alamat = Negara("Bandung")
# print(alamat.kota)

# alamat.desaKu("Isola")  # ^^^^
# print(alamat.desa)
# print(alamat.dusun)
""" --------> End of experiment 3 <-------- """
""" * * *---> Start of experiment 4 <---* * * """
import cv2
import cvzone

cap = cv2.VideoCapture("D:/AI_CCTV/.runs/videos/0624.mp4")
table_bg = cv2.imread(".runs/images/OL1.png", cv2.IMREAD_UNCHANGED)

new_width = 1000
aspect_ratio = table_bg.shape[1] / table_bg.shape[0]
new_height = int(new_width / aspect_ratio)
table_bg = cv2.resize(table_bg, (new_width, new_height))

while True:
    _, frame = cap.read()
    frame_with_overlay = cvzone.overlayPNG(frame, table_bg, (50, 50))
    cv2.imshow("Video with Overlay", frame_with_overlay)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

""" --------> End of experiment 4 <-------- """
