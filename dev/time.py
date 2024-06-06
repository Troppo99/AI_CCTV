# import time

# waktu_awal = time.time()

# while True:
#     waktu = time.time()
#     print(f"waktu : {round(waktu-waktu_awal):02}")


# from datetime import timedelta
# def format_time(seconds):
#     return str(timedelta(seconds=int(seconds)))

# seconds = 20000
# print(type(format_time(seconds)))

from datetime import datetime

waktu = datetime.now().strftime("%Y/%m/%d %H:%M:%S")
print(type(waktu))
