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
import cv2
import random
from sklearn.neighbors import KDTree


def gaussian_pyramid(image,levels):
	assert np.power(2,levels)<=image.shape[0] and np.power(2,levels)<=image.shape[1]
	layer = image.copy()
	arr = [layer]
	for i in range(levels):
		layer = cv2.pyrDown(layer)
		arr.append(layer)
	return arr

def laplacian_pyramid(image,levels):
	assert np.power(2,levels)<=image.shape[0] and np.power(2,levels)<=image.shape[1]
	layer = image.copy()
	gp = gaussian_pyramid(layer,levels)
	lp = []
	for i in range(1, levels+1):
		expanded_image = cv2.pyrUp(gp[i])
		j = i-1
		laplacian = cv2.subtract(gp[j],expanded_image)
		lp.append(laplacian)
	lp.append(gp[-1])
	return lp,gp

def reconstructed(lp):
	levels = len(lp)
	corrected_image = lp[-1]
	for i in range(levels-2,-1,-1):
		expanded_image = cv2.pyrUp(corrected_image)
		corrected_image = cv2.add(expanded_image,lp[i])
	return corrected_image


def nearest_neighbor(pyramid,level):
    rows,cols = pyramid[level].shape 
    vectors = []
    for i in range(rows):
        for j in range(cols):
            neighbor = getNeighbourhood(pyramid,level,[i,j])
            if(len(vectors)==0):
                vectors_arr = np.array([neighbor]).reshape((1,neighbor.shape[0]))
                vectors.append(1)
            else:
                vectors_arr = np.vstack([vectors_arr,neighbor.reshape((1,neighbor.shape[0]))])
    return KDTree(vectors_arr),vectors_arr.shape[0]

def randomRow2Map(copyFrom, copyTo,rowsToCopy):
    rowsEx, colsEx= np.shape(copyFrom)
    rows, cols= np.shape(copyTo)
    rand_r = random.randint(ceil((rowsEx - rowsToCopy)/4), int(rowsEx - rowsToCopy))
    rand_c = random.randint(0, int(colsEx/2))
    copyTo[0:rowsToCopy,0:int(colsEx/2)] = copyFrom[rand_r:rand_r+rowsToCopy, rand_c:rand_c+int(colsEx/2)]
    return copyTo

def multi_resolution_texture_synthesis(img,output_shape,levels):
    output_img = np.zeros(output_shape)
    input_gp = gaussian_pyramid(img,levels)
    output_gp = gaussian_pyramid(output_img,levels)
    level = len(output_gp)
    randomRow2Map(input_gp[-1], output_gp[-1],2) 
    for i in range(level-1,-1,-1):
        curr_level_inp = input_gp[i]
        curr_level_out = output_gp[i]
        rows,cols = curr_level_out.shape
        tree,n_vectors = nearest_neighbor(input_gp,i)
        for r in range(rows):
            for c in range(cols):
                C = findBestMatch(input_gp,output_gp,i,[r,c],tree,k = min(n_vectors,1))
                output_gp[i][r,c] = input_gp[i][C[0],C[1]]
        plt.imshow(output_gp[i],cmap='gray')
        plt.show()
        print("done")
            

def padding(img, pad):
    return np.lib.pad(img, pad, "constant", constant_values=0)

def getSingleLevelNeighbourhood(pyramid,level, coord, kernel, mode):
    
    if mode=='parent':
        coord[0] = floor(coord[0]/2)
        coord[1] = floor(coord[1]/2)

    half_kernel = floor(kernel / 2)
    padded = padding(pyramid[level], half_kernel)
    shifted_row = coord[0] + half_kernel
    shifted_col = coord[1] + half_kernel
    row_start = shifted_row - half_kernel
    row_end = shifted_row + half_kernel + 1
    col_start = shifted_col - half_kernel
    col_end = shifted_col + half_kernel + 1
    padded = padded[row_start:row_end, col_start:col_end]
    if mode=='parent':
        return padded.reshape(np.shape(padded)[0]*np.shape(padded)[1], 1)
    if mode=='child': 
        return padded.reshape(np.shape(padded)[0]*np.shape(padded)[1], 1)[0:floor(kernel*kernel/2), :]

def getNeighbourhood(pyramid, pyramidLevel, coord, mirror_hor=False, mirror_vert=False):
    Nchild = getSingleLevelNeighbourhood(pyramid, pyramidLevel, coord, 5, mode='child')
    if pyramidLevel+1<=len(pyramid)-1:
        Nparent = getSingleLevelNeighbourhood(pyramid, pyramidLevel+1, coord, 3, mode='parent')
    else:
        Nparent = np.zeros((3 * 3, 1))
    N = np.concatenate((Nchild, Nparent), axis=0)
    return N

def findBestMatch(G_inp,G_out,curr_level,coord,tree,k):
    N = getNeighbourhood(G_out,curr_level,coord)
    dist,ind = tree.query([N.reshape(-1)],k=k)
    dist = dist[0]
    ind = ind[0]
    coord = ind[0]
    r = floor(coord/G_inp[curr_level].shape[0])
    c = coord - r*G_inp[curr_level].shape[1]
    return [r,c]



if __name__ == "__main__":
    img_path = "brodatz/D95.gif"
    img = color.rgb2gray(io.imread(img_path))
    if(img.shape[0]>400):
        img = rescale(img,0.15,anti_aliasing=False)
        plt.imshow(img,cmap='gray')
        plt.show()
    multi_resolution_texture_synthesis(img,(200,200),3)
