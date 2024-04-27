import pandas as pd
import numpy as np

from glob import glob
import cv2
import matplotlib.pylab as plt

cat_files = glob('../PYTHONPROJECT/training_set/cats/*.jpg')
dog_files = glob('../PYTHONPROJECT/training_set/dogs/*.jpg')

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

fig, axs = plt.subplots(1,2,figsize=(8,5))
axs[0].imshow(img_cv2)
axs[1].imshow(img_mpl)
axs[0].axis('off')
axs[1].axis('off')
axs[0].set_title('CV Image')
axs[1].set_title('Matplotlib Image')
plt.show()

# img_cv2_rgb = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)
# fig, ax= plt.subplots()
# ax.imshow(img_cv2_rgb)
# ax.axis('off')
# plt.show()