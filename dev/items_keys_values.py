# Membuat dictionary
my_dict = {"name": "Alice", "age": 25, "city": "Wonderland"}

# Menggunakan .values() untuk mendapatkan semua nilai
values = my_dict.values()

for i in my_dict.items():
    print(i)
for i in my_dict.keys():
    print(i)
for i in my_dict.values():
    print(i)

# Mencetak objek pandangan nilai
print(values)  # Output: dict_values(['Alice', 25, 'Wonderland'])

# Mengubah nilai dalam dictionary
my_dict["age"] = 26

# Pandangan nilai akan mencerminkan perubahan
print(values)  # Output: dict_values(['Alice', 26, 'Wonderland'])
