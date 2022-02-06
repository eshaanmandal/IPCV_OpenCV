from turtle import color
import cv2 as cv
import math
import matplotlib.pyplot as plt
import numpy as np
import sys
plt.rcParams['figure.figsize']=[6,6]


# numberofbins = ceil( (maximumvalue - minimumvalue) / binwidth 
def preprocess(image):
    '''Applies the required preprocessing for the image
    -> converts RGB to Grayscale
    -> reduce the image resolution to make computations little bit easy for the computer 
    '''
    if len(image.shape) == 3:
        image = cv.cvtColor(image,cv.COLOR_BGR2GRAY)
    if image.shape[0]*image.shape[1] > 10_000:
        image = cv.resize(image, (100,100))
    return image

def histogram(img, bins=256):
    '''Calculates the histogram for the image, bins can be provided by the user or it assumes bins = 256 (8-bit image)'''
    hist_array = [0 for _ in range(bins)]
    for i in img:
        for j in i:
            hist_array[math.ceil(int(j)*(bins-1)/255)]+=1
    return  hist_array

nbins = input("BINS (for default press d): ")
nbins =  256 if nbins == 'd' else int(nbins)
img_path = 'fox.jpeg'


img = cv.imread(img_path)
img = preprocess(img)
 
h = histogram(img, bins=nbins)
plt.bar(range(nbins),h,color='green')
plt.show()

