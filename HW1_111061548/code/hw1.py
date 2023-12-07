# Before trying to construct hybrid images, it is suggested that you
# implement my_imfilter.py and then debug it using hw1_test_filtering.py
import os
import cv2
import numpy as np
import matplotlib
matplotlib.use('TkAgg')         # pycharm bug
import matplotlib.pyplot as plt

from my_imfilter import my_imfilter
from visualize_hybrid_image import visualize_hybrid_image
from normalize import normalize
from gauss2D import gauss2D

def main():
    """ function to create hybrid images """
    # read images and convert to floating point format
    main_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    image1 = cv2.cvtColor(cv2.imread(os.path.join(main_path, 'data', 'dog.bmp')), cv2.COLOR_BGR2RGB)
    image2 = cv2.cvtColor(cv2.imread(os.path.join(main_path, 'data', 'cat.bmp')), cv2.COLOR_BGR2RGB)
    image1 = image1.astype(np.float32)/255
    image2 = image2.astype(np.float32)/255

    # Several additional test cases are provided for you, but feel free to make
    # your own (you'll need to align the images in a photo editor such as
    # Photoshop). The hybrid images will differ depending on which image you
    # assign as image1 (which will provide the low frequencies) and which image
    # you asign as image2 (which will provide the high frequencies)

    # ===============================================
    # === Filtering and Hybrid Image construction ===
    # ===============================================
    cutoff_frequency = 7
    # This is the standard deviation, in pixels, of the
    # Gaussian blur that will remove the high frequencies from one image and 
    # remove the low frequencies from another image (by subtracting a blurred
    # version from the original version). You will want to tune this for every
    # image pair to get the best results.
    gaussian_filter = gauss2D(shape=(cutoff_frequency*4+1,cutoff_frequency*4+1), sigma = cutoff_frequency)


    # =======================================================================
    # TODO: Use my_imfilter create 'low_frequencies' and                    
    # 'high_frequencies' and then combine them to create 'hybrid_image'     
    # =======================================================================
    # =======================================================================
    # Remove the high frequencies from image1 by blurring it. The amount of 
    # blur that works best will vary with different image pairs             
    # =======================================================================
    low_frequencies = my_imfilter(image1, gaussian_filter)
    low_frequencies = normalize(low_frequencies)
    print('low_frequencies done !')

    # ==========================================================================
    # Remove the low frequencies from image2. The easiest way to do this is to 
    # subtract a blurred version of image2 from the original version of image2.
    # This will give you an image centered at zero with negative values.       
    # ==========================================================================
    high_frequencies = image2 - my_imfilter(image2, gaussian_filter)
    high_frequencies = high_frequencies + 0.5
    high_frequencies = normalize(high_frequencies)
    print('high_frequencies done !')

    # ==========================================================================
    # Combine the high frequencies and low frequencies                         
    # ==========================================================================
    hybrid_image = low_frequencies + high_frequencies
    hybrid_image = normalize(hybrid_image)
    print('hybrid_image done !')


    ### Visualize and save outputs ###
    plt.figure("Low Frequencies")
    plt.imshow(low_frequencies)
    plt.figure("High Frequencies")
    plt.imshow(high_frequencies)
    hybrid_image_scales = visualize_hybrid_image(hybrid_image)
    hybrid_image_scales = normalize(hybrid_image_scales)
    plt.figure("Hybrid Image")
    plt.imshow(hybrid_image_scales)
    plt.imsave(os.path.join(main_path, 'results', 'low_frequencies.png'), low_frequencies, dpi=95)
    plt.imsave(os.path.join(main_path, 'results', 'high_frequencies.png'), high_frequencies, dpi=95)
    plt.imsave(os.path.join(main_path, 'results', 'hybrid_image.png'), hybrid_image, dpi=95)
    plt.imsave(os.path.join(main_path, 'results', 'hybrid_image_scales.png'), hybrid_image_scales, dpi=95)

    plt.show()

def celebrities(cutoff_frequency):
    main_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    image1 = cv2.cvtColor(cv2.imread(os.path.join(main_path, 'data', 'einstein.bmp')), cv2.COLOR_BGR2RGB)
    image2 = cv2.cvtColor(cv2.imread(os.path.join(main_path, 'data', 'marilyn.bmp')), cv2.COLOR_BGR2RGB)
    image1 = image1.astype(np.float32)/255
    image2 = image2.astype(np.float32)/255
    gaussian_filter = gauss2D(shape=(cutoff_frequency*4+1, cutoff_frequency*4+1), sigma=cutoff_frequency)
    low_frequencies = my_imfilter(image1, gaussian_filter)
    low_frequencies = normalize(low_frequencies)
    print('low_frequencies done !')
    high_frequencies = image2 - my_imfilter(image2, gaussian_filter)
    high_frequencies = high_frequencies + 0.5
    high_frequencies = normalize(high_frequencies)
    print('high_frequencies done !')
    hybrid_image = low_frequencies + high_frequencies
    hybrid_image = normalize(hybrid_image)
    print('hybrid_image done !')
    hybrid_image_scales = visualize_hybrid_image(hybrid_image)
    hybrid_image_scales = normalize(hybrid_image_scales)
    plt.imsave(os.path.join(main_path, 'results', 'hybrid_celebrities.png'), hybrid_image, dpi=95)
    plt.imsave(os.path.join(main_path, 'results', 'hybrid_celebrities_scales.png'), hybrid_image_scales, dpi=95)

def flying_objects(cutoff_frequency):
    main_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    image1 = cv2.cvtColor(cv2.imread(os.path.join(main_path, 'data', 'bird.bmp')), cv2.COLOR_BGR2RGB)
    image2 = cv2.cvtColor(cv2.imread(os.path.join(main_path, 'data', 'plane.bmp')), cv2.COLOR_BGR2RGB)
    image1 = image1.astype(np.float32)/255
    image2 = image2.astype(np.float32)/255
    gaussian_filter = gauss2D(shape=(cutoff_frequency*4+1, cutoff_frequency*4+1), sigma=cutoff_frequency)
    low_frequencies = my_imfilter(image1, gaussian_filter)
    low_frequencies = normalize(low_frequencies)
    print('low_frequencies done !')
    high_frequencies = image2 - my_imfilter(image2, gaussian_filter)
    high_frequencies = high_frequencies + 0.5
    high_frequencies = normalize(high_frequencies)
    print('high_frequencies done !')
    hybrid_image = low_frequencies + high_frequencies
    hybrid_image = normalize(hybrid_image)
    print('hybrid_image done !')
    hybrid_image_scales = visualize_hybrid_image(hybrid_image)
    hybrid_image_scales = normalize(hybrid_image_scales)
    plt.imsave(os.path.join(main_path, 'results', 'hybrid_flying_objects.png'), hybrid_image, dpi=95)
    plt.imsave(os.path.join(main_path, 'results', 'hybrid_flying_objects_scales.png'), hybrid_image_scales, dpi=95)

def ocean(cutoff_frequency):
    main_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    image1 = cv2.cvtColor(cv2.imread(os.path.join(main_path, 'data', 'fish.bmp')), cv2.COLOR_BGR2RGB)
    image2 = cv2.cvtColor(cv2.imread(os.path.join(main_path, 'data', 'submarine.bmp')), cv2.COLOR_BGR2RGB)
    image1 = image1.astype(np.float32)/255
    image2 = image2.astype(np.float32)/255
    gaussian_filter = gauss2D(shape=(cutoff_frequency*4+1, cutoff_frequency*4+1), sigma=cutoff_frequency)
    low_frequencies = my_imfilter(image1, gaussian_filter)
    low_frequencies = normalize(low_frequencies)
    print('low_frequencies done !')
    high_frequencies = image2 - my_imfilter(image2, gaussian_filter)
    high_frequencies = high_frequencies + 0.5
    high_frequencies = normalize(high_frequencies)
    print('high_frequencies done !')
    hybrid_image = low_frequencies + high_frequencies
    hybrid_image = normalize(hybrid_image)
    print('hybrid_image done !')
    hybrid_image_scales = visualize_hybrid_image(hybrid_image)
    hybrid_image_scales = normalize(hybrid_image_scales)
    plt.imsave(os.path.join(main_path, 'results', 'hybrid_ocean.png'), hybrid_image, dpi=95)
    plt.imsave(os.path.join(main_path, 'results', 'hybrid_ocean_scales.png'), hybrid_image_scales, dpi=95)

def two_wheels(cutoff_frequency):
    main_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    image1 = cv2.cvtColor(cv2.imread(os.path.join(main_path, 'data', 'bicycle.bmp')), cv2.COLOR_BGR2RGB)
    image2 = cv2.cvtColor(cv2.imread(os.path.join(main_path, 'data', 'motorcycle.bmp')), cv2.COLOR_BGR2RGB)
    image1 = image1.astype(np.float32)/255
    image2 = image2.astype(np.float32)/255
    gaussian_filter = gauss2D(shape=(cutoff_frequency*4+1, cutoff_frequency*4+1), sigma=cutoff_frequency)
    low_frequencies = my_imfilter(image1, gaussian_filter)
    low_frequencies = normalize(low_frequencies)
    print('low_frequencies done !')
    high_frequencies = image2 - my_imfilter(image2, gaussian_filter)
    high_frequencies = high_frequencies + 0.5
    high_frequencies = normalize(high_frequencies)
    print('high_frequencies done !')
    hybrid_image = low_frequencies + high_frequencies
    hybrid_image = normalize(hybrid_image)
    print('hybrid_image done !')
    hybrid_image_scales = visualize_hybrid_image(hybrid_image)
    hybrid_image_scales = normalize(hybrid_image_scales)
    plt.imsave(os.path.join(main_path, 'results', 'hybrid_two_wheels.png'), hybrid_image, dpi=95)
    plt.imsave(os.path.join(main_path, 'results', 'hybrid_two_wheels_scales.png'), hybrid_image_scales, dpi=95)

def dragonball(cutoff_frequency):
    main_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    image1 = cv2.cvtColor(cv2.imread(os.path.join(main_path, 'data', 'Son_Goten.bmp')), cv2.COLOR_BGR2RGB)
    image2 = cv2.cvtColor(cv2.imread(os.path.join(main_path, 'data', 'Trunks.bmp')), cv2.COLOR_BGR2RGB)
    image1 = image1.astype(np.float32)/255
    image2 = image2.astype(np.float32)/255
    gaussian_filter = gauss2D(shape=(cutoff_frequency*4+1, cutoff_frequency*4+1), sigma=cutoff_frequency)
    low_frequencies = my_imfilter(image1, gaussian_filter)
    low_frequencies = normalize(low_frequencies)
    print('low_frequencies done !')
    high_frequencies = image2 - my_imfilter(image2, gaussian_filter)
    high_frequencies = high_frequencies + 0.5
    high_frequencies = normalize(high_frequencies)
    print('high_frequencies done !')
    hybrid_image = low_frequencies + high_frequencies
    hybrid_image = normalize(hybrid_image)
    print('hybrid_image done !')
    hybrid_image_scales = visualize_hybrid_image(hybrid_image)
    hybrid_image_scales = normalize(hybrid_image_scales)
    plt.imsave(os.path.join(main_path, 'results', 'hybrid_dragonball.png'), hybrid_image, dpi=95)
    plt.imsave(os.path.join(main_path, 'results', 'hybrid_dragonball_scales.png'), hybrid_image_scales, dpi=95)

def classmate(cutoff_frequency):
    main_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    image1 = cv2.cvtColor(cv2.imread(os.path.join(main_path, 'data', 'lin.bmp')), cv2.COLOR_BGR2RGB)
    image2 = cv2.cvtColor(cv2.imread(os.path.join(main_path, 'data', 'wang.bmp')), cv2.COLOR_BGR2RGB)
    image1 = image1.astype(np.float32)/255
    image2 = image2.astype(np.float32)/255
    gaussian_filter = gauss2D(shape=(cutoff_frequency*4+1, cutoff_frequency*4+1), sigma=cutoff_frequency)
    low_frequencies = my_imfilter(image1, gaussian_filter)
    low_frequencies = normalize(low_frequencies)
    print('low_frequencies done !')
    high_frequencies = image2 - my_imfilter(image2, gaussian_filter)
    high_frequencies = high_frequencies + 0.5
    high_frequencies = normalize(high_frequencies)
    print('high_frequencies done !')
    hybrid_image = low_frequencies + high_frequencies
    hybrid_image = normalize(hybrid_image)
    print('hybrid_image done !')
    hybrid_image_scales = visualize_hybrid_image(hybrid_image)
    hybrid_image_scales = normalize(hybrid_image_scales)
    plt.imsave(os.path.join(main_path, 'results', 'hybrid_classmate.png'), hybrid_image, dpi=95)
    plt.imsave(os.path.join(main_path, 'results', 'hybrid_classmate_scales.png'), hybrid_image_scales, dpi=95)

if __name__ == '__main__':
    main()
    #celebrities(3)
    #flying_objects(5)
    #ocean(5)
    #two_wheels(5)
    #dragonball(8)
    #classmate(4)



