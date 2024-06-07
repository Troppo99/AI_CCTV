class Buku:
    def __init__(self, judul, penulis, tahun_terbit):
        # Parameter: judul, penulis, tahun_terbit
        # Atribut: self.judul, self.penulis, self.tahun_terbit
        self.judul = judul  # Atribut
        self.penulis = penulis  # Atribut
        self.tahun_terbit = tahun_terbit  # Atribut

    def deskripsi(self):
        # self adalah parameter khusus yang merujuk pada instance objek saat ini
        # Tidak ada parameter lain di sini
        return f"{self.judul} oleh {self.penulis}, diterbitkan tahun {self.tahun_terbit}"

    def update_tahun_terbit(self, tahun_baru):
        # self adalah parameter khusus, tahun_baru adalah parameter biasa
        self.tahun_terbit = tahun_baru  # Mengupdate atribut tahun_terbit dengan nilai baru


# Membuat objek dari kelas Buku
buku1 = Buku("Harry Potter", "J.K. Rowling", 1997)
buku2 = Buku("Lord of the Rings", "J.R.R. Tolkien", 1954)

# Menggunakan metode deskripsi pada objek
print(buku1.deskripsi())  # Output: Harry Potter oleh J.K. Rowling, diterbitkan tahun 1997
print(buku2.deskripsi())  # Output: Lord of the Rings oleh J.R.R. Tolkien, diterbitkan tahun 1954

# Menggunakan metode update_tahun_terbit pada objek
buku1.update_tahun_terbit(1998)
print(buku1.deskripsi())  # Output: Harry Potter oleh J.K. Rowling, diterbitkan tahun 1998
