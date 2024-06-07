class Orang:
    def __init__(self, nama, umur):
        self.nama = nama
        self.umur = umur

    def perkenalan(self):
        return f"Halo, nama saya {self.nama} dan saya berumur {self.umur} tahun."


# Membuat objek dari kelas Orang
orang1 = Orang("Budi", 30)
orang2 = Orang("Ani", 25)

# Menggunakan metode perkenalan pada objek
print(orang1.perkenalan())  # Output: Halo, nama saya Budi dan saya berumur 30 tahun.
print(orang2.perkenalan())  # Output: Halo, nama saya Ani dan saya berumur 25 tahun.

