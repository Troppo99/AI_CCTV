import pandas as pd
import mysql.connector

# Koneksi ke database MySQL
conn = mysql.connector.connect(
    host="localhost",
    user="robot",
    password="robot123",
    database="report_ai_cctv",
    port=3307
)

# Query untuk mengambil data dari tabel
query = "SELECT * FROM activity_log"

# Membaca data ke dalam DataFrame pandas
df = pd.read_sql(query, conn)

# Mengekspor data ke file CSV
df.to_csv("D:/NWR27/AI_CCTV/.runs/data/output1.csv", index=False)

# Menutup koneksi
conn.close()
