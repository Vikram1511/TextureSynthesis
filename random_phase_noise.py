import numpy as np
import matplotlib.pyplot as plt 
import sys
import random
from skimage import color,io


def smoothing_func(alpha,m,n):
    kernel = np.zeros((m,))
    for i in range(0,m+1):
      
        if(i<int(m*alpha) or i>(m-int(alpha*m))):
            print(i)
            kernel[i-1] = np.exp(-1/(1-((2*i/alpha)-1)**2))
        else:
            kernel[i-1] = 1
    print(kernel)

def rpn(img,output_shape):
    m,n=img.shape
    output_img = np.zeros(output_shape)
    output_img.fill(img.mean())
    out_h = output_shape[0]
    out_w = output_shape[1]
    new_img = img
    output_img[out_h//2 - m//2:out_h//2+m//2,out_w//2 - n//2:out_w//2+n//2] = new_img
    random_phase_ = random_phase1(out_h,out_w)
    fft = np.fft.fft2(output_img)
    k_shift = np.fft.fftshift(fft)
    k_shift = np.real(k_shift)*np.exp(1j*random_phase_)
    orig_img = np.fft.ifft2(np.fft.ifftshift(k_shift))
    # fft_mag = np.real(periodic_component)
    # fft_phase = np.imag(periodic_component)

    # random_uniform_phase = random_phase1(m,n)
    # fft_phase_2 = fft_phase+random_uniform_phase
    # modified_period_comp = fft_mag*np.exp(1j*random_uniform_phase)
    # output_img[out_h//2 - m//2:out_h//2+m//2,out_w//2 - n//2:out_w//2+n//2] = modified_period_comp
    plt.imshow(np.absolute(orig_img),cmap='gray')
    plt.imsave("output.jpg",np.absolute(orig_img),cmap='gray')
    plt.show()
    

def random_phase1(shapex,shapey):
    mat1 = np.random.uniform(low=0,high=np.pi,size=(shapex//2,shapey//2))
    mat2 = np.random.uniform(low=0,high=np.pi,size=(shapex//2,shapey//2))
    mat_flip = np.hstack([mat2,mat1])
    mat_flip2 = np.hstack([np.flip(mat1),np.flip(mat2)])
    phase = np.vstack([mat_flip2,mat_flip])
    phase[int(phase.shape[0]//2),int(phase.shape[1]//2)]=0
    return phase

def random_phase(shapex,shapey):
    half_x = shapex//2
    half_y = shapey//2
    x_even = shapex%2
    y_even = shapey%2
    sx  = int((shapex/2)+1)
    output_img = np.zeros((shapex,shapey),dtype=complex)
    invN = float(1/float(shapex*shapey))
    sign=None
    for y in range(0,shapey):
        for x in range(0,shapex):
            if(((x==0) | ((x_even==0) & (x==half_x))) & ((y==0) | ((y_even==0) & (y==half_x)))):
                if((x==0)&(y==0)):
                    sign=1
                else:
                    num = random.uniform(0,1)
                    if(num<0.5):
                        sign=1
                    else:
                        sign = -1
                output_img[y][x] = sign*invN + 0j
            elif(((x==0) | ((x_even==0) & (x==half_x)))& (y>half_y)):
                addres_pixel  =(shapey-y) +x 
                output_img[y][x] = output_img[addres_pixel][y]
            else:
                theta =   (2*random.uniform(0,1)-1)*np.pi 
                costheta = float(np.cos(theta))
                sintheta = float(np.sin(theta))
                output_img[y][x] = (costheta*invN) + (1j*sintheta*invN)
    return output_img
                


if __name__ == "__main__":
    img_path = sys.argv[1]
    img = color.rgb2gray(io.imread(img_path))
    rpn(img,(1000,1000))