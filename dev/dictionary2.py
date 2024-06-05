# perpus = {'name' : {'asli': 'nana', 'palsu': 'nwr'},'umur' : 22, 'hobby' : 'Swimming'}

# print(perpus['name']['palsu'])

# ***********************************************************************************************************************************
def find_key_by_value(dictionary, target_value):
    for key, value in dictionary.items():
        if value == target_value:
            return key
    return None


# Contoh penggunaan fungsi
perpus = {"name": "Nana", "umur": 22, "hobby": "Swimming"}

# Mencari kunci berdasarkan value
value_to_find = "Nana"
found_key = find_key_by_value(perpus, value_to_find)

if found_key:
    print(f"Kunci untuk value '{value_to_find}' adalah '{found_key}'")
else:
    print(f"Value '{value_to_find}' tidak ditemukan dalam kamus")
