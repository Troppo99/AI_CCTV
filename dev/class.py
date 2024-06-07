# Definisi kelas dasar
class Kendaraan:
    def __init__(self, merk, model, tahun):
        self.merk = merk
        self.model = model
        self.tahun = tahun

    def deskripsi(self):
        return f"{self.tahun} {self.merk} {self.model}"

    def suara(self):
        return "Vroom"


# Kelas turunan
class Mobil(Kendaraan):
    def __init__(self, merk, model, tahun, jumlah_pintu):
        super().__init__(merk, model, tahun)
        self.jumlah_pintu = jumlah_pintu

    def deskripsi(self):
        return f"{self.tahun} {self.merk} {self.model} dengan {self.jumlah_pintu} pintu"


# Kelas turunan lain
class Motor(Kendaraan):
    def suara(self):
        return "Ngeeng"


# Membuat objek dari kelas-kelas tersebut
mobil_saya = Mobil("Toyota", "Camry", 2020, 4)
motor_saya = Motor("Honda", "CBR", 2018)

# Menggunakan metode-metode dari objek
print(mobil_saya.deskripsi())  # Output: 2020 Toyota Camry dengan 4 pintu
print(mobil_saya.suara())  # Output: Vroom
print(motor_saya.deskripsi())  # Output: 2018 Honda CBR
print(motor_saya.suara())  # Output: Ngeeng
