import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


def rotate_image(image, angle):
  image_center = tuple(np.array(image.shape[1::-1]) / 2)
  rot_mat = cv.getRotationMatrix2D(image_center, angle, 1.0)
  result = cv.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv.INTER_LINEAR)
  return result

img = cv.imread('cuboid.jpeg')
rotated_img = rotate_image(img, 45)

plt.imshow(rotated_img)
plt.show()
