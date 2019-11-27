import numpy as np
from skimage import io, morphology, exposure
from random import randint
from math import ceil, floor
import matplotlib.pyplot as plt
import time
import sys
from skimage import color
from skimage.transform import rescale
import os
seed_size = 3
k_size = 17
half_window = 8
error_threshold_max = 0.1

def load_sample_image(image_path):
    sample_image = color.rgb2gray(io.imread(image_path))
    if(sample_image.shape[0]>400):
        sample_image = rescale(sample_image,0.15,anti_aliasing=False)
    sample_image = sample_image/255.0
    print(sample_image.shape)
    return sample_image

def gauss2D(shape, sigma):
    m, n = [(ss-1.)/2. for ss in shape]
    y, x = np.ogrid[-m:m+1,-n:n+1]
    h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
    h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h

def allpossible_patches(img, k_size):
    r, c = np.shape(img)
    slide_x =r - (k_size-1);
    slide_y =c - (k_size-1);
    patches_array = np.zeros((slide_x*slide_y,k_size,k_size))
    for i in range(slide_x):
        for j in range(slide_y):
            patches_array[i*slide_y+j] = img[i:i+k_size,j:j+k_size]         
    return patches_array

def find_matches(template,image_window, valid_mask, gauss_mask):
    template = np.reshape(template, k_size*k_size)
    gauss_mask = np.reshape(gauss_mask, k_size*k_size)
    valid_mask = np.reshape(valid_mask, k_size*k_size)
    total_weight = np.sum(np.multiply(gauss_mask, valid_mask))
    image_window = image_window.reshape((image_window.shape[0],image_window.shape[1]*image_window.shape[2]))
    template = template.reshape((1,template.shape[0]))
    distance = (image_window-template)**2
    ssd = np.sum((distance*gauss_mask*valid_mask) / total_weight, axis=1)
    min_error = min(ssd)
    mid = int(((2 * (half_window) + 1) ** 2) / 2);
    res = []
    image_window = image_window.tolist()
    for i, err in enumerate(ssd):
        if err <= min_error*(1+error_threshold_max):
            res.append([err,image_window[i][mid]])
    return res

def texture_synt(sample_img,output_shape,save_path):
    os.makedirs(save_path[:-1])
    start_time = time.time()
    error_threshold_max= 0.3

    img_row, img_col = np.shape(sample_img)
    sigma = k_size/ 6.4;

    patches_all = allpossible_patches(sample_img,k_size)
    # interval = ceil(number_pixel/100)
    out_image = np.zeros(output_shape)
    m,n = output_shape[0],output_shape[1]
    seed_size = 3
    random_row = randint(0, img_row - seed_size)
    random_col = randint(0, img_col - seed_size)
    seed = sample_img[random_row:random_row + seed_size, random_col:random_col + seed_size]
    out_image[floor(m/ 2):floor(m / 2) + seed_size,floor(n / 2):floor(n / 2) + seed_size] = seed

    number_filled = seed_size * seed_size
    filled_map = np.zeros(output_shape)
    filled_map[floor(m/ 2):floor(m/ 2) + seed_size,floor(n / 2):floor(n/ 2) + seed_size] = np.ones((seed_size, seed_size))

    g_mask = gauss2D((k_size,k_size), sigma=sigma)

    out_image_padded = np.lib.pad(out_image, half_window , 'constant', constant_values=0)
    filled_map_padded = np.lib.pad(filled_map, half_window, 'constant', constant_values=0)
    number_pixel = m*n
    while number_filled < number_pixel:
        progress = False
        candidate_pixel_row, candidate_pixel_col = np.nonzero(morphology.binary_dilation(filled_map) - filled_map)
        neighborHood = []
        for i in range(len(candidate_pixel_row)):
            pixel_row = candidate_pixel_row[i]
            pixel_col = candidate_pixel_col[i]
            neighborHood.append(np.sum(filled_map[pixel_row - half_window : pixel_row + half_window + 1,
                                       pixel_col - half_window : pixel_col + half_window + 1]))

        order = np.argsort(-np.array(neighborHood, dtype=int))
        # print order
        for x, i in enumerate(order): #range(len(candidate_pixel_row)):
            pixel_row = candidate_pixel_row[i]
            pixel_col = candidate_pixel_col[i]
            template_img = out_image_padded[pixel_row - half_window + half_window:pixel_row + half_window + half_window + 1,
                pixel_col - half_window + half_window:pixel_col + half_window + half_window + 1]
            
            best_match = find_matches(template_img,patches_all,
                filled_map_padded[pixel_row - half_window + half_window:pixel_row + half_window + half_window + 1,
                pixel_col - half_window + half_window:pixel_col + half_window+ half_window + 1],
                g_mask)
            pick = randint(0, len(best_match)-1)
            # print "Matches= ", len(best_match)
            if best_match[pick][0]<=error_threshold_max:
                out_image_padded[half_window+pixel_row][half_window+pixel_col] = best_match[pick][1]
                out_image[pixel_row][pixel_col]=best_match[pick][1]
                filled_map_padded[half_window+pixel_row][half_window+pixel_col] = 1
                filled_map[pixel_row][pixel_col]=1
                number_filled+=1
                if number_filled % 2000 == 0:
                    plt.imsave(save_path+"out"+str(number_filled)+".jpg",out_image,cmap='gray')
                progress = True
        if not progress:
            error_threshold_max *= 1.1
    print("time taken",time.time()-start_time)
    plt.imsave(save_path+"output.jpg", out_image,cmap='gray')

# if __name__ == "__main__":
#     img_path = sys.argv[1]
#     sample_img = load_sample_image(img_path)
#     texture_synt(sample_img,(200,200))