import pandas as pd
import numpy as np
from glob import glob
import cv2
import matplotlib.pylab as plt
plt.style.use('ggplot')

'''Reading in images'''
cat_files = glob('../AI_CCTV/Learn01/training_set/cats/*.jpg')
dog_files = glob('../AI_CCTV/Learn01/training_set/dogs/*.jpg')
# print(cat_files[20])
img_mpl = plt.imread(cat_files[20])
img_cv2 = cv2.imread(cat_files[20])

# print(img_mpl)
# print(type(img_mpl))
# print(type(img_cv2))
# print(img_mpl.shape,", ",img_cv2.shape)
# print(img_mpl.max)
# print(img_mpl.flatten())

# print(pd.Series(img_cv2.flatten()).plot(kind='hist', bins=50,title='Distribution of Pixel Values'))
# plt.show()

# print(type(plt.subplots(1,3,figsize=(5,5))))

'''Display Images'''
# fig, ax = plt.subplots(figsize=(10, 10))
# ax.imshow(img_mpl)
# ax.axis('off')
# plt.show()

# fig, ax= plt.subplots(1,3,figsize=(5,5))
# ax[0].imshow(img_mpl[:,:,0], cmap='Reds')
# ax[1].imshow(img_mpl[:,:,1], cmap='Greens')
# ax[2].imshow(img_mpl[:,:,2], cmap='Blues')
# ax[0].axis('off')
# ax[1].axis('off')
# ax[2].axis('off')
# ax[0].set_title('Red Channel')
# ax[1].set_title('Green Channel')
# ax[2].set_title('Blue Channel')
# plt.show()

# fig, axs = plt.subplots(1,2,figsize=(8,5))
# axs[0].imshow(img_cv2)
# axs[1].imshow(img_mpl)
# axs[0].axis('off')
# axs[1].axis('off')
# axs[0].set_title('CV Image')
# axs[1].set_title('Matplotlib Image')
# plt.show()

# img_cv2_rgb = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)
# fig, ax= plt.subplots()
# ax.imshow(img_cv2_rgb)
# ax.axis('off')
# plt.show()
# print(type(plt.subplots()))

'''Image Manipulation'''
# img = plt.imread(dog_files[4])
# fig, ax = plt.subplots(figsize=(5, 5))
# ax.imshow(img)
# ax.axis('off')
# plt.show()

# img = plt.imread(dog_files[4])
# img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
# fig, ax = plt.subplots(figsize=(5, 5))
# ax.imshow(img_gray, cmap='Greys')
# ax.axis('off')
# ax.set_title('Grey Image')
# plt.show()

'''Resizing and Scall=ing'''
# img = plt.imread(dog_files[4])
# img_resized = cv2.resize(img, None, fx=0.25, fy=0.25)
# fig, ax = plt.subplots(figsize=(5, 5))
# ax.imshow(img_resized)
# ax.axis('off')
# plt.show()

# # Different Size
# img = plt.imread(dog_files[4])
# img_resize = cv2.resize(img, (100, 200))
# fig, ax = plt.subplots(figsize=(5, 5))
# ax.imshow(img_resize)
# ax.axis('off')
# plt.show()

# img = plt.imread(dog_files[4])
# img_resize = cv2.resize(img, (5000, 5000), interpolation = cv2.INTER_CUBIC)
# fig, ax = plt.subplots(figsize=(5, 5))
# ax.imshow(img_resize)
# ax.axis('off')
# plt.show()