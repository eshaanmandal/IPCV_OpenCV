from configparser import Interpolation
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

def convolve(X, F):
    # height and width of the image
    X_height = X.shape[0]
    X_width = X.shape[1]
    
    # height and width of the filter
    F_height = F.shape[0]
    F_width = F.shape[1]
    
    H = (F_height - 1) // 2
    W = (F_width - 1) // 2
    
    #output numpy matrix with height and width
    out = np.zeros((X_height, X_width))
    #iterate over all the pixel of image X
    for i in np.arange(H, X_height-H):
        for j in np.arange(W, X_width-W):
            sum = 0
            #iterate over the filter
            for k in np.arange(-H, H+1):
                for l in np.arange(-W, W+1):
                    #get the corresponding value from image and filter
                    a = X[i+k, j+l]
                    w = F[H+k, W+l]
                    sum += (w * a)
            out[i,j] = sum
    #return convolution  
    return out

my_image = cv.resize(cv.cvtColor(cv.imread('use_this.jpg'), cv.COLOR_BGR2GRAY), (512, 512))
sketch = cv.resize(cv.cvtColor(cv.imread('guts.jpg'), cv.COLOR_BGR2GRAY), (512, 512))
black_image = cv.resize(cv.cvtColor(cv.imread('black.jpg'), cv.COLOR_BGR2GRAY), (512, 512))
tree = cv.resize(cv.cvtColor(cv.imread('tree.jpg'), cv.COLOR_BGR2GRAY), (512, 512))


# sobel filter kernels
Gx = np.array([[-1, 0, 1],[-2, 0, 2],[-1, 0, 1]])
Gy = np.array([[-1, -2, -1],[0, 0, 0],[1, 2, 1]])

# Prewitt  filter kernels
Gx_p = np.array([[1, 0, -1],[1, 0, -1],[1, 0, -1]])
Gy_p = np.array([[1, 1, 1],[0, 0, 0],[-1, -1, -1]])

# Convolution operation
sob_my_image_x = convolve(my_image, Gx) / 8.0
sob_my_image_y = convolve(my_image, Gy) / 8.0

sob_sketch_x = convolve(sketch, Gx) / 8.0
sob_sketch_y = convolve(sketch, Gy) / 8.0

sob_black_image_x = convolve(black_image, Gx) / 8.0
sob_black_image_y = convolve(black_image, Gy) / 8.0


sob_tree_x = convolve(tree, Gx) / 8.0
sob_tree_y = convolve(tree, Gy) / 8.0

#calculate the gradient magnitude of vectors
sob_out_my_image = np.sqrt(np.power(sob_my_image_x, 2) + np.power(sob_my_image_y, 2))
sob_out_sketch = np.sqrt(np.power(sob_sketch_x, 2) + np.power(sob_sketch_y, 2))
sob_out_black_image = np.sqrt(np.power(sob_black_image_x, 2) + np.power(sob_black_image_y, 2))
sob_out_tree = np.sqrt(np.power(sob_tree_x, 2) + np.power(sob_tree_y, 2))
# mapping values from 0 to 255

sob_out_my_image = (sob_out_my_image / np.max(sob_out_my_image)) * 255
sob_out_sketch = (sob_out_sketch / np.max(sob_out_sketch)) * 255
sob_out_black_image = (sob_out_black_image) * 255
sob_out_tree = (sob_out_tree / np.max(sob_out_tree)) * 255


prew_my_image_x = convolve(my_image, Gx_p) / 8.0
prew_my_image_y = convolve(my_image, Gy_p) / 8.0

prew_sketch_x = convolve(sketch, Gx_p) / 8.0
prew_sketch_y = convolve(sketch, Gy_p) / 8.0

prew_black_image_x = convolve(black_image, Gx_p) / 8.0
prew_black_image_y = convolve(black_image, Gy_p) / 8.0


prew_tree_x = convolve(tree, Gx_p) / 8.0
prew_tree_y = convolve(tree, Gy_p) / 8.0

#calculate the gradient magnitude of vectors
prew_out_my_image = np.sqrt(np.power(prew_my_image_x, 2) + np.power(prew_my_image_y, 2))
prew_out_sketch = np.sqrt(np.power(prew_sketch_x, 2) + np.power(prew_sketch_y, 2))
prew_out_black_image = np.sqrt(np.power(prew_black_image_x, 2) + np.power(prew_black_image_y, 2))
prew_out_tree = np.sqrt(np.power(prew_tree_x, 2) + np.power(prew_tree_y, 2))
# mapping values from 0 to 255

prew_out_my_image = (prew_out_my_image / np.max(prew_out_my_image)) * 255
prew_out_sketch = (prew_out_sketch / np.max(prew_out_sketch)) * 255
prew_out_black_image = (prew_out_black_image) * 255
prew_out_tree = (prew_out_tree / np.max(prew_out_tree)) * 255


# plotting the outputs


_, axs = plt.subplots(3,4)
axs[0][0].imshow(my_image, cmap='gray')
axs[0][0].set_title("My image")
axs[0][1].imshow(black_image, cmap='gray')
axs[0][1].set_title("Black Image")
axs[0][2].imshow(sketch, cmap='gray')
axs[0][2].set_title("Sketch")
axs[0][3].imshow(tree, cmap='gray')
axs[0][3].set_title("Tree")

axs[1][0].imshow(sob_out_my_image, cmap='gray')
axs[1][0].set_title("My image sobel")
axs[1][1].imshow(sob_out_black_image, cmap='gray')
axs[1][1].set_title("Black image sobel")
axs[1][2].imshow(sob_out_sketch, cmap='gray')
axs[1][2].set_title("Sketch sobel")
axs[1][3].imshow(sob_out_tree, cmap='gray')
axs[1][3].set_title("Tree sobel")

axs[2][0].imshow(prew_out_my_image, cmap='gray')
axs[2][0].set_title("My image prewitt")
axs[2][1].imshow(prew_out_black_image, cmap='gray')
axs[2][1].set_title("Black image prewitt")
axs[2][2].imshow(prew_out_sketch, cmap='gray')
axs[2][2].set_title("Sketch prewitt")
axs[2][3].imshow(prew_out_tree, cmap='gray')
axs[2][3].set_title("Tree prewitt")

plt.show()

