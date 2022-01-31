import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

def split_n_shuffle(image):
    '''
    Divides the image into 4 parts
    and arranges it according to the question
    '''
    row, col = image.shape
    p1, p2, p3, p4 = image[0:int(row/2), 0:int(col/2)], image[0:int(row/2),int(col/2):col+1], image[int(row/2):row+1, 0:int(col/2)], image[int(row/2):row+1, int(col/2):col+1]
    print(p1.shape, p2.shape, p3.shape, p4.shape)
    shuffled_image = np.zeros_like(image)
    shuffled_image[0:int(row/2), 0:int(col/2)] = p4
    shuffled_image[0:int(row/2),int(col/2):col+1] = p1
    shuffled_image[int(row/2):row+1, 0:int(col/2)] = p2
    shuffled_image[int(row/2):row+1, int(col/2):col+1] = p3

    return shuffled_image
def preprocess(image):
    '''Applies the required preprocessing for the image
    -> converts RGB to Grayscale
    -> Converts Odd numbers of rows and columns to even number of rows of columns
    '''
    if len(image.shape) == 3:
        image = cv.cvtColor(image,cv.COLOR_BGR2GRAY)
    if image.shape[0] % 2 != 0:
        image = np.delete(image, -1, axis=0)
    if image.shape[1] % 2 != 0:
        image = np.delete(image, -1, axis=1)
    return image


img_path = './bird.jpeg'
image = preprocess(cv.imread(img_path))
shuffled_image = split_n_shuffle(image)

images = [image, shuffled_image]
captions = ['Original Image', 'Reorganized Image']

plt.figure(figsize=(8,8))
for i in range(2):
    plt.subplot(1,2,i+1)
    plt.imshow(images[i],cmap=plt.cm.gray)
    plt.title(captions[i])
    plt.axis('off')

plt.show()
