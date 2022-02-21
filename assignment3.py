import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [8,8]



if __name__ == "__main__":
    "Read an image and convert it to grayscale"
    img = cv.imread('low_contrast.png')
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    "Applying histogram eqaulization"
    equalized = cv.equalizeHist(img)


    "Plotting the results"
    plt.subplot(1,3,2)
    plt.hist(img.flatten(), 256, (0,256))
    plt.title('Histogram')
    

    plt.subplot(1,3,3)
    plt.hist(equalized.flatten(), 256, (0,256))
    plt.title('Contrast equalized histogram')
    

    plt.subplot(1,3,1)
    plt.imshow(img, cmap=plt.cm.gray)
    plt.title('Original Image')
    plt.axis('off')

    plt.show()





    

    

