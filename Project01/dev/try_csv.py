import csv

# Data yang akan ditulis ke file CSV
data = [
    ["Nama", "Usia", "Kota"],
    ["Alice", 30, "New York"],
    ["Bob", 25, "Los Angeles"],
    ["Charlie", 35, "Chicago"],
]

# Nama file CSV
filename = "Project01/dev/output.csv"

# Menulis data ke file CSV
with open(filename, mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerows(data)

print(f"Data telah berhasil ditulis ke {filename}")
