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


class Negara:
    def __init__(self, city="Majalengka"):
        self.kota = city
        # self.desaKu() # ini ada hubungannya dengan ^^^^

    def desaKu(self, desa="Mekarmulya", dusun="Cisahang"):
        self.desa = desa
        self.dusun = dusun


alamat = Negara("Bandung")
print(alamat.kota)

alamat.desaKu("Isola")  # ^^^^
print(alamat.desa)
print(alamat.dusun)
""" --------> End of experiment 3 <-------- """
