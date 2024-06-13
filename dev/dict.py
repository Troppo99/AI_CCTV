my_dict = {"satu": 1, "dua": 2, 3: 4, 4: 5}

element = "tujuh"
print(my_dict)

if element in my_dict.keys():
    keys = list(my_dict.keys())
    position = keys.index(element)
    print(f'"{element}" ditemukan sebagai key ke {position+1}')
if element in my_dict.values():
    keys = list(my_dict.values())
    position = keys.index(element)
    print(f'"{element}" ditemukan sebagai value ke {position+1}')
else:
    my_dict["tujuh"] = 7
    print(my_dict)
    # print(f'"{element}" tidak ditemukan sebagai kunci maupun nilai')
