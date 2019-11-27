#%%
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import math
import sys, os
import logging
from scipy.signal import convolve2d
from skimage.transform import resize
from skimage import color,io
import cv2
class SteerablePyramid:
	def __init__(self, image, n, k, out_path):
			self.INP_IMAGE=image
			# self.IMAGE_ARRAY = np.asarray(image, dtype='complex')
			# self.IMAGE_NAME = image_name
			#		self.OUT_PATH = out_path # path to the directory for saving images.
			self.OUT_PATH = out_path + '/{}' # path to the directory for saving images.
			self.orientation = k
			if(n>np.log2(np.min(np.array([self.INP_IMAGE.shape[0], self.INP_IMAGE.shape[1]])))-1):
				raise ValueError('illegal depth: {}'.format(str(n)))
			self.depth = n
			self.ALPHAK = 2.**(self.orientation-1) * math.factorial(self.orientation-1)/np.sqrt(self.orientation * float(math.factorial(2.*(self.orientation-1))))
			self.RES_at_levels = []
			for i in range(0, self.depth):
				self.RES_at_levels.append( (int(self.INP_IMAGE.shape[0]/2**i), int(self.INP_IMAGE.shape[1]/2**i)) )
			self.H0 = {'frequency_dom':None, 'spatial_dom':None}
			self.L0 = {'frequency_dom':None, 'spatial_dom':None}
			self.LR = {'frequency_dom':None, 'spatial_dom':None}
			self.BND = []
			self.LOW = []

			self.RADIAL, self.ANGULAR = self.get_polar_coordinates()
			self.H0_FILT = self.get_h0_filter()
			self.L0_FILT = self.get_l0_filter()
			self.L_FILT = self.get_l_filter()
			self.H_FILT = self.get_h_filter()
			self.B_FILT =  self.get_b_filters()

	def get_polar_coordinates(self):
			pol = []
			ang = []
			for i in range(0, self.depth):
				res_x = self.RES_at_levels[i][0]
				res_y = self.RES_at_levels[i][1]
				x = np.linspace(-np.pi, np.pi, num = res_x, endpoint = False)
				y = np.linspace(-np.pi, np.pi, num = res_y, endpoint = False)
				radial_ = np.zeros((x.shape[0],y.shape[0]))
				yy, xx= np.meshgrid(x,y)
				radial_ = np.sqrt((xx)**2 + (yy)**2)

				angular= np.zeros((x.shape[0],y.shape[0]))
				angular[np.where((yy == 0) & (xx < 0))] = np.pi
				ids = np.where((yy != 0) | (xx >= 0))
				angular[ids] = np.arctan2(yy[ids], xx[ids])
				pol.append(radial_)
				ang.append(angular)
			return pol, ang


	def get_h0_filter(self):
		resx,resy = self.INP_IMAGE.shape[0],self.INP_IMAGE.shape[1]
		filter_ = np.zeros((resx,resy))
		filter_[np.where(self.RADIAL[0] >= np.pi)] = 1
		filter_[np.where(self.RADIAL[0] < np.pi/2.)] = 0
		_ind = np.where(( np.pi/2.<self.RADIAL[0]) & (self.RADIAL[0] < np.pi))
		filter_[_ind] = np.cos(np.pi/2. * np.log2( self.RADIAL[0][_ind]/np.pi) )
		return filter_

	def get_l0_filter(self):
		resx,resy = self.INP_IMAGE.shape[0],self.INP_IMAGE.shape[1]
		filter_ = np.zeros((resx,resy))
		filter_[np.where(self.RADIAL[0] >= np.pi)] = 0
		filter_[np.where(self.RADIAL[0] <= np.pi/2.)] = 1
		_ind = np.where((self.RADIAL[0] > np.pi/2.) & (self.RADIAL[0] < np.pi))
		filter_[_ind] = np.cos(np.pi/2. * np.log2(2. * self.RADIAL[0][_ind]/np.pi))
		return filter_
    
	def get_l_filter(self):
		filters = []
		for i in range(0, self.depth):
			resx,resy = self.RES_at_levels[i][0],self.RES_at_levels[i][1]
			filter_ = np.zeros((resx,resy))
			filter_[np.where(self.RADIAL[i] >= np.pi/2.)] = 0
			filter_[np.where(self.RADIAL[i] <= np.pi/4.)] = 1
			_ind = np.where((self.RADIAL[i] > np.pi/4.) & (self.RADIAL[i] < np.pi/2.))
			filter_[_ind] = np.cos(np.pi/2. * np.log2(4. * self.RADIAL[i][_ind]/np.pi))
			filters.append(filter_)
		return filters
	def get_h_filter(self):
		filters = []
		for i in range(0, self.depth):
			resx,resy = self.RES_at_levels[i][0],self.RES_at_levels[i][1]
			filter_ = np.zeros((resx,resy))
			filter_[np.where(self.RADIAL[i] >= np.pi/2.)] = 1
			filter_[np.where(self.RADIAL[i] <= np.pi/4.)] = 0
			_ind = np.where((self.RADIAL[i] > np.pi/4.) & (self.RADIAL[i] < np.pi/2.))
			filter_[_ind] = np.cos(np.pi/2. * np.log2(2.*self.RADIAL[i][_ind]/np.pi))

			filters.append(filter_)		
		return filters	
	def get_b_filters(self):
		filter_bank = []
		for i in range(0, self.depth):
			resx,resy = self.RES_at_levels[i][0],self.RES_at_levels[i][1]
			filters = []
			for k in range(self.orientation):
				filter_= np.zeros((resx,resy), dtype=complex)
				th1= self.ANGULAR[i].copy()
				th2= self.ANGULAR[i].copy()

				th1[np.where(self.ANGULAR[i] - k*np.pi/self.orientation < -np.pi)] += 2.*np.pi
				th1[np.where(self.ANGULAR[i] - k*np.pi/self.orientation > np.pi)] -= 2.*np.pi
				ind_ = np.where(np.absolute(th1 - k*np.pi/self.orientation) <= np.pi/2.)
				filter_[ind_] = self.ALPHAK * (np.cos(th1[ind_] - k*np.pi/self.orientation))**(self.orientation-1)
				th2[np.where(self.ANGULAR[i] + (self.orientation-k)*np.pi/self.orientation < -np.pi)] += 2.*np.pi
				th2[np.where(self.ANGULAR[i] + (self.orientation-k)*np.pi/self.orientation > np.pi)] -= 2.*np.pi
				ind_ = np.where(np.absolute(th2 + (self.orientation-k) * np.pi/self.orientation) <= np.pi/2.)
				filter_[ind_] = self.ALPHAK * (np.cos(th2[ind_]+ (self.orientation-k) * np.pi/self.orientation))**(self.orientation-1)

				filter_= self.H_FILT[i] * filter_
				filters.append(filter_.copy())

			filter_bank.append(filters)
		return filter_bank

	def make_steerable_pyramids(self):
		ft = np.fft.fft2(self.INP_IMAGE)
		_ft = np.fft.fftshift(ft)
		frquency_dom_img = _ft * self.H0_FILT
		f_ishift = np.fft.ifftshift(frquency_dom_img)
		spatial_dom_img = np.fft.ifft2(f_ishift)

		self.H0['frequency_dom'] = frquency_dom_img.copy()
		self.H0['spatial_dom'] = spatial_dom_img.copy()
        
		frquency_dom_img = _ft * self.L0_FILT
		f_ishift = np.fft.ifftshift(frquency_dom_img)
		spatial_dom_img = np.fft.ifft2(f_ishift)
		self.L0['frequency_dom'] = frquency_dom_img.copy()
		self.L0['spatial_dom'] = spatial_dom_img.copy()

		_last = frquency_dom_img
		for i in range(self.depth):
			curr_res_BND = []
			curr_level_b_filters = len(self.B_FILT[i])
			for j in range(curr_level_b_filters):
				curr_orientation = self.B_FILT[i][j]
				curr_dict = {'frequency_dom':None, 'spatial_dom':None}
				frquency_dom_img = _last * curr_orientation
				f_ishift = np.fft.ifftshift(frquency_dom_img)
				spatial_dom_img = np.fft.ifft2(f_ishift)

				curr_dict['frequency_dom'] = frquency_dom_img
				curr_dict['spatial_dom'] = spatial_dom_img
				curr_res_BND.append(curr_dict)

			self.BND.append(curr_res_BND.copy())
			l1 = _last * self.L_FILT[i]
			down_image = l1[l1.shape[0]//4:3*(l1.shape[0]//4),l1.shape[1]//4:3*(l1.shape[1]//4)]
			f_ishift = np.fft.ifftshift(down_image)
			img_back = np.fft.ifft2(f_ishift)
			self.LOW.append({'frequency_dom':down_image, 'spatial_dom':img_back})
			_last = down_image

		self.LR['frequency_dom'] = _last.copy()
		self.LR['spatial_dom'] = img_back.copy()
		return None
		
	def reconstruction_image(self):
		residual_img = self.LR['frequency_dom']
		for level in range(self.depth-1,-1,-1):
			m,n = residual_img.shape
			upsampled_residual = np.zeros((2*m,2*n),dtype=complex)
			upsampled_residual[m//2:3*(m//2),n//2:3*(n//2)] = residual_img
			residual_img = upsampled_residual

			residual_img = residual_img*self.L_FILT[level]
			for band_filt in range(len(self.B_FILT[level])):
				residual_img  = residual_img+(self.BND[level][band_filt]['frequency_dom']*self.B_FILT[level][band_filt])
		image = residual_img*self.L0_FILT+self.H0['frequency_dom']+self.H0_FILT
		return image


def hist_match(source_img,template_img):
	m,n = source_img.shape
	source_img = source_img.ravel()
	template_img =template_img.ravel()
	unqiue_val,ind,counts = np.unique(source_img,return_counts=True,return_inverse=True)
	unique_temp,ind_temp,counts_temp = np.unique(template_img,return_counts=True,return_inverse=True)
	cum_sum = np.cumsum(counts).astype(np.float64)
	cum_sum = cum_sum/cum_sum[-1]
	temp_cumsum = np.cumsum(counts_temp).astype(np.float64)
	temp_cumsum = temp_cumsum/temp_cumsum[-1]
	interp_t_values = np.interp(cum_sum, temp_cumsum, unique_temp)

	return interp_t_values[ind].reshape((m,n))

def getFFTspectrum(img,a):
    K = np.fft.fft2(img,norm='ortho')
    Kshift = np.fft.fftshift(K)
    magnitude_spec =a*np.log(np.abs(Kshift)) 
    return magnitude_spec,K

def texture_mapping(texture_img,output_shape,iteration_num):
		texture = texture_img.copy()
		noise_img  = np.random.rand(output_shape[0],output_shape[1])
		noise_img= np.fft.ifft2(np.fft.ifftshift(np.fft.fftshift(np.fft.fft2(noise_img))*np.real(np.fft.fftshift(np.fft.fft2(texture)))))
		noise_img = hist_match(noise_img,texture)
		
		# fft_spec_img,_ = getFFTspectrum(texture,60)
		# fft_spec_noise,_ = getFFTspectrum(noise_img,60)

		sp1 = SteerablePyramid(texture,4,4,"/")
		sp1.make_steerable_pyramids()
		# plt.imshow(fft_spec_noise)
		# plt.show()
		# plt.imshow(fft_spec_img)
		# plt.show()
		for i in range(iteration_num):
			sp2 = SteerablePyramid(noise_img,4,4,"/")
			sp2.make_steerable_pyramids()
			level = len(sp2.BND)-1
			while(level>=0):
				for i in range(len(sp2.BND[level])):
					# subbnd_texture =np.absolute(sp1.BND[level][i]['spatial_dom'])
					# subbnd_noise = np.absolute(sp2.BND[level][i]['spatial_dom'])
					# sp2.BND[level][i]['spatial_dom'] = hist_match(subbnd_noise,subbnd_texture)
					# sp2.BND[level][i]['frequency_dom'] = np.fft.fftshift(np.fft.fft2(sp2.BND[level][i]['spatial_dom']))

					subband_txt_real  = np.real(sp1.BND[level][i]['frequency_dom'])
					subband_txt_img = np.imag(sp1.BND[level][i]['frequency_dom'])
					subband_noise = np.real(sp2.BND[level][i]['frequency_dom'])
					subband_noise_imag = np.imag(sp2.BND[level][i]['frequency_dom'])
					subband_noise_real = hist_match(subband_noise,subband_txt_real)
					subband_noise_imag = hist_match(subband_noise_imag,subband_txt_img)
					sp2.BND[level][i]['frequency_dom'] = subband_noise_real + 1j*subband_noise_imag
				level=level-1
			# sp2.H0['frequency_dom'] = np.fft.fftshift(np.fft.fft2(hist_match(np.absolute(sp2.H0['spatial_dom']),np.absolute(sp1.H0['spatial_dom']))))
			# sp2.LR['frequency_dom'] = np.fft.fftshift(np.fft.fft2(hist_match(np.absolute(sp2.LR['spatial_dom']),np.absolute(sp1.LR['spatial_dom']))))
			h_freq = sp2.H0['frequency_dom']
			lr_freq = sp2.LR['frequency_dom']
			h_freq_real = np.real(h_freq)
			h_freq_imag = np.imag(h_freq)
			lr_freq_imag = np.imag(lr_freq)
			lr_freq_real = np.real(lr_freq)

			lr_freq_real  = hist_match(lr_freq_real,np.real(sp1.LR['frequency_dom']))
			lr_freq_imag = hist_match(lr_freq_imag,np.imag(sp1.LR['frequency_dom']))
			h_freq_real  = hist_match(h_freq_real,np.real(sp1.H0['frequency_dom']))
			h_freq_imag = hist_match(h_freq_imag,np.imag(sp1.H0['frequency_dom']))

			sp2.H0['frequency_dom'] = h_freq_real + 1j*h_freq_imag
			sp2.LR['frequency_dom'] = lr_freq_real + 1j*lr_freq_imag

			# sp2.H0['frequency_dom'] = hist_match(sp2.H0['frequency_dom'],sp1.H0['frequency_dom'])
			# sp2.LR['frequency_dom'] = hist_match(sp2.LR['frequency_dom'],sp1.LR['frequency_dom'])
			noise_img = hist_match(np.real(np.fft.ifft2(np.fft.ifftshift(sp2.reconstruction_image()))),texture)
			sp2=None 
		return noise_img


#%%
if __name__ == "__main__":
	img_path = sys.argv[1]
	img = color.rgb2gray(io.imread(img_path))
	plt.imsave('input.jpg',img,cmap='gray')
	m,n = img.shape
	# img = img[m//2-64:m//2+64,m//2-64:m//2+64]
	img_ = texture_mapping(img,(m,n),5)
	plt.imshow(img_,cmap='gray')
	plt.imsave("output.jpg",img_,cmap='gray')
	plt.show()


# %%
