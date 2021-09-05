import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import matplotlib.image as mpimg
import cv2

im1 = cv2.imread('im_1.jpg')
im2 = cv2.imread('im_2.jpg')
im1 = cv2.resize(im1, (2048, 1080), interpolation=cv2.INTER_LINEAR)
im2 = cv2.resize(im2, (2048, 1080), interpolation=cv2.INTER_LINEAR)
# plt.imshow(im1)
cv2.imwrite('im_1.jpg', im1)
cv2.imwrite('im_2.jpg', im2)
for i in range(1, 10):
    lam = i * 0.1
    im_mixup = (im1 * lam + im2 * (1 - lam)).astype(np.uint8)
    image_array = np.array(im_mixup)
    image_output = Image.fromarray(image_array)
    image_output.save("im_mix_" + str(i) + ".jpg")
    # cv2.imwrite('im_mix_', im_mixup)
    # plt.subplot(3, 3, i)
    # plt.axis('off')
    # plt.imshow(im_mixup)
    # plt.savefig('im_mixup_'+str(i))
# plt.show()
