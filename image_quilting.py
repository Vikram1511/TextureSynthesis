import numpy as np 
from skimage import color,io 
import matplotlib.pyplot as plt 
import random 
import sys
import os
from math import ceil

def scan_blocks(image,blocksize,overlap_size):
    m,n = image.shape
    tiles=[]
    for w in range(0,m-overlap_size,blocksize):
        for h in range(0,n-overlap_size,blocksize):
            block = image[w:w+blocksize,h:h+blocksize]
            if(block.shape[0]==blocksize and block.shape[1]==blocksize):
                tiles.append(block)
    return tiles
        

def find_tile_match_LT(refblock_left,refblock_top,overlap_size,blocksize,block_list):
    errors = dict()
    for i in range(len(block_list)):
        rms_val = ((refblock_left[:,blocksize-overlap_size:blocksize] - block_list[i][:,:overlap_size])**2).mean()
        rms_val = rms_val + ((refblock_top[:overlap_size,:] - block_list[i][blocksize-overlap_size:,:])**2).mean()
        errors[i]=rms_val

    errors_val = sorted(errors.items(), key = lambda kv:(kv[1], kv[0]))
    size = len(list(errors.keys()))
    top_10 = int(size*0.1)
    if(top_10<1):
        choosen=errors_val[0][0]
    else:
        top_keys = [errors_val[i][0] for i in range(top_10)]

        choosen = random.choice(top_keys)
    choosen_tile = block_list[choosen]
    return choosen_tile


def find_tile_match(refblock,overlap_size,blocksize,block_list,horiz=False,vert=False):
    errors = dict()
    for i in range(len(block_list)):
        if(horiz==True):
            rms_val = ((refblock[:,blocksize-overlap_size:blocksize] - block_list[i][:,:overlap_size])**2).mean()
        if(vert==True):
            rms_val = ((refblock[:overlap_size,:] - block_list[i][blocksize-overlap_size:,:])**2).mean()
        errors[i] = rms_val
    errors_val = sorted(errors.items(), key = lambda kv:(kv[1], kv[0]))
    size = len(list(errors.keys()))
    top_10 = int(size*0.1)
    if(top_10<1):
        choosen=errors_val[0][0]
    else:
        top_keys = [errors_val[i][0] for i in range(top_10)]

        choosen = random.choice(top_keys)
    choosen_tile = block_list[choosen]
    return choosen_tile


def get_min_error_surface_vertical(block1, block2, blocksize, overlap):
    # err_surface = ((block1[:, -overlap:] - block2[:, :overlap])**2)
    # cost_matrix = np.zeros(err_surface.shape)
    # for i in range(err_surface.shape[1]):
    #     cost_matrix[0,i] = err_surface[0,i]
    
    # min_index=[]
    # min_index.append(np.argmin(cost_matrix[0,:]))
    # for r in range(1,err_surface.shape[0]-1):
    #     for c in range(1,err_surface.shape[1]-1):
    #         cost_matrix[r,c] = err_surface[r,c]+min(cost_matrix[r-1,c-1],cost_matrix[r-1,c],cost_matrix[r-1,c+1])
    
    #     min_index.append(np.argmin(cost_matrix[r,:]))
    # mask = np.zeros((blocksize, blocksize))
    # for i in range(len(min_index)):
    #     mask[i, :min_index[i]+1] = 1

    # resBlock = np.zeros(block1.shape)
    # resBlock[:, :overlap] = block1[:, -overlap:]
    # resBlock = resBlock*mask + block2*(1-mask)
    # return resBlock,mask

    err_surface = ((block1[:, -overlap:] - block2[:, :overlap])**2)
    min_path = []
    curr_row = err_surface[0].tolist()
    cost_mat = []
    cost_mat.append(curr_row)
    for r in range(1, err_surface.shape[0]):
        e = [np.float('inf')] + cost_mat[-1] + [np.float('inf')]
        e = np.array([e[:-2], e[1:-1], e[2:]])
        min_arr = e.min(0)
        min_ind = e.argmin(0) - 1
        min_path.append(min_ind)
        E = err_surface[r] + min_arr
        cost_mat.append(list(E))

    path = []
    minArg = np.argmin(cost_mat[-1])
    path.append(minArg)
    for idx in min_path[::-1]:
        minArg = minArg + idx[minArg]
        path.append(minArg)

    
    path = path[::-1]
    mask = np.zeros((blocksize, blocksize))
    for i in range(len(path)):
        mask[i, :path[i]+1] = 1
    resBlock = np.zeros(block1.shape)
    resBlock[:, :overlap] = block1[:, -overlap:]
    resBlock = resBlock*mask + block2*(1-mask)
    return resBlock,mask

def get_min_error_surface_horizontal(block1, block2, blocksize, overlap):
	resBlock,mask = get_min_error_surface_vertical(np.rot90(block1), np.rot90(block2), blocksize, overlap)
	return np.rot90(resBlock, 3),np.rot90(mask,3)


def get_min_error_surface_both(refblock_left,refblock_top,mainblock,blocksize,overlap):
    res_block_left,mask1 = get_min_error_surface_vertical(refblock_left,mainblock,blocksize,overlap)
    res_block_top,mask2 = get_min_error_surface_horizontal(refblock_top,mainblock,blocksize,overlap)
    mask2[:overlap, :overlap] = np.maximum(mask2[:overlap, :overlap] - mask1[:overlap, :overlap], 0)

    resBlock = np.zeros(mainblock.shape)
    resBlock[:, :overlap] = mask1[:, :overlap]*refblock_left[:, -overlap:]
    resBlock[:overlap, :] = resBlock[:overlap, :] + mask2[:overlap, :]*refblock_top[-overlap:, :]
    resBlock = resBlock + (1-np.maximum(mask1, mask2))*mainblock
    return resBlock
    

def img_quilting(input_img,out_shape,overlap_size,blocksize,save_path):
    m,n = input_img.shape
    nH = int(ceil((out_shape[0] - blocksize)*1.0/(blocksize - overlap_size)))
    nW = int(ceil((out_shape[1] - blocksize)*1.0/(blocksize - overlap_size)))
    output_img = np.zeros(((blocksize + nH*(blocksize - overlap_size)), (blocksize + nW*(blocksize - overlap_size))))
    outM,outN = output_img.shape
    block_scanned = scan_blocks(input_img,blocksize,overlap_size)
    randm = np.random.randint(len(block_scanned))
    start = block_scanned[randm]
    # sample_block = input_img[randm:randm+blocksize,randn:randn+blocksize]
    output_img[:blocksize,:blocksize] = start

    #filling firstrow

    for i in range(blocksize-overlap_size,outN-overlap_size,blocksize-overlap_size):
        refblock = output_img[:blocksize,i+overlap_size-blocksize:i+overlap_size]
        res_block = find_tile_match(refblock,overlap_size,blocksize,block_scanned,horiz=True)
        res_block,_ = get_min_error_surface_vertical(refblock,res_block,blocksize,overlap_size)
        output_img[:blocksize,i:i+blocksize] = res_block
    
    #filling first column

    for i in range(blocksize-overlap_size,outM-overlap_size,blocksize-overlap_size):
        refblock = output_img[i+overlap_size-blocksize:i+overlap_size,:blocksize]
        res_block = find_tile_match(refblock,overlap_size,blocksize,block_scanned,vert=True)
        res_block,_ = get_min_error_surface_horizontal(refblock,res_block,blocksize,overlap_size)
        output_img[i:i+blocksize,:blocksize] = res_block

    #filling rest tiles

    for j in range(blocksize-overlap_size,outM-overlap_size,blocksize-overlap_size):
        for i in range(blocksize-overlap_size,outN-overlap_size,blocksize-overlap_size):
            refblock_left = output_img[j:j+blocksize,i+overlap_size-blocksize:i+overlap_size]
            refblock_top = output_img[j+overlap_size-blocksize:j+overlap_size,i:i+blocksize]
            res_block = find_tile_match_LT(refblock_left,refblock_top,overlap_size,blocksize,block_scanned)
            res_block = get_min_error_surface_both(refblock_left,refblock_top,res_block,blocksize,overlap_size)
            output_img[j:j+blocksize,i:i+blocksize] = res_block

    plt.imsave(save_path,output_img,cmap='gray')
    




    plt.imshow(output_img,cmap='gray')
    plt.show()

def load_sample_image(image_path):
    sample_image = color.rgb2gray(io.imread(image_path))
    sample_image = sample_image/255.0
    print(sample_image.shape)
    return sample_image

if __name__ == "__main__":
    img_path = sys.argv[1]
    block_size = int(sys.argv[2])
    overlap_size=block_size//6
    output_shape=(int(sys.argv[3]),int(sys.argv[4]))
    img = load_sample_image(img_path)
    img_name = img_path.split("/")[-1]
    plt.imsave("textures/"+img_name[:-4]+"gray.jpg",img,cmap='gray')
    save_path = sys.argv[5]
    img_quilting(img,output_shape,overlap_size,block_size,save_path)





    

        
        

