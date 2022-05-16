import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


# Reading an image from the current working directory
img = cv.imread('cuboid.jpeg')
# converting BGR to RGB
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
img_copy = np.copy(img)

# Converting the image from BGR colorspace to Grayscale
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# Using Canny edge detector to find lines
edges = cv.Canny(gray, 50, 150, apertureSize=3)

# Using probabilistic Hough transform
lines = cv.HoughLinesP(edges,1,np.pi/180,threshold=50,minLineLength=2, maxLineGap=40)

# print(lines.shape)
for i in range(lines.shape[0]):
    x1, y1, x2, y2 = lines[i,0]
    # Draws a line on the image, the lines are highlighted in red color
    cv.line(img_copy,(x1,y1), (x2,y2), (255, 0, 0),2)

plt.imsave('lines.jpg', img_copy)

plt.subplot(1,3,1)
plt.imshow(img)
plt.title("Original Image")
plt.axis('off')

plt.subplot(1,3,2)
plt.imshow(edges, cmap='gray')
plt.title("Edges of the image")
plt.axis('off')

plt.subplot(1,3,3)
plt.imshow(img_copy)
plt.title("Image with lines highlighted")
plt.axis('off')

plt.show()