import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

plt.rcParams['figure.figsize']=[8,8]

if __name__=="__main__":
    img = cv.resize(cv.cvtColor(cv.imread('use_this.jpg'), cv.COLOR_BGR2GRAY), (1000, 1000))
    black_image = np.full((1000, 1000), 255)
    white_image = np.full((1000, 1000), 0)

    fft_img = np.fft.fftshift(np.fft.fft2(img))
    fft_black = np.fft.fftshift(np.fft.fft2(black_image))
    fft_white = np.fft.fftshift(np.fft.fft2(white_image))

    # plotting the magnitudes of the fourier coeffecients (as it is a complex number)
    magn_img = np.abs(fft_img)
    magn_black = np.abs(fft_black)
    magn_white = np.abs(fft_white)

    _, axs = plt.subplots(1,4)

    # plot the diagrams
    axs[0].imshow(img, cmap="gray")
    axs[0].set_title("Image")

    axs[1].imshow(np.log(magn_img+0.1), cmap='gray')
    axs[1].set_title("FFT of image")

    axs[2].imshow(np.log(magn_black+0.1), cmap='gray')
    axs[2].set_title("FFT of black image")

    axs[3].imshow(np.log(magn_white+0.1), cmap='gray')
    axs[3].set_title("FFT of white image")
 
    plt.show()
    