# t=0,5,"yuyu",False
# print(type(t[0]))

# t=False,True
# print(t)

# web = (123, 'Petani Kode', 'https://www.petanikode.com')

# # lalu kita ingin potong agar ditampilkan
# # dari indeks nomer 1 sampai 2
# print(web[1:2])
'''A'''
# import cv2
# cap = cv2.VideoCapture(1)

# # myImage = cv2.imread('training_set/cats/cat.1.jpg')

# while True:
#     ret, myImage = cap.read()
#     myImage_edit = cv2.cvtColor(myImage, cv2.COLOR_BGR2GRAY)
#     myImage_RGB = cv2.cvtColor(myImage,cv2.COLOR_BGR2RGB)
#     myImage_LUV = cv2.cvtColor(myImage,cv2.COLOR_BGR2LUV)

#     cv2.imshow('myCat',myImage)
#     cv2.imshow('myCat_GRAY',myImage_edit)
#     cv2.imshow('myCat_RGB',myImage_RGB)
#     cv2.imshow('myCat_LUV',myImage_LUV)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.rel
# cv2.destroyAllWindows()
'''A'''
import cv2

# Read an image in RGB format
rgb_image = cv2.imread('training_set/cats/cat.1.jpg')

# Convert RGB image to BGR
bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)

# Display the original RGB image
cv2.imshow('Original RGB Image', rgb_image)

# Display the converted BGR image
cv2.imshow('Converted BGR Image', bgr_image)

# Wait for a key press and then close the windows
cv2.waitKey(0)
cv2.destroyAllWindows()
