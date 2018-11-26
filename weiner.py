import cv2
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage.filters as fi
from scipy.signal import convolve2d
from scipy import signal
from sklearn.metrics import mean_squared_error
import math
import sys
import warnings
warnings.filterwarnings("ignore")

#====================== To Add Blur and Noise ========================
def blur(kernlen, nsig, gray):	# To add gaussian blur to an image
    inp = np.zeros((kernlen, kernlen)) 
    inp[kernlen//2, kernlen//2] = 1
    B = fi.gaussian_filter(inp, nsig) # Gaussian Kernal of size kernlen*kernlen 
    
    A = gray
    C2 = convolve2d(A, B, mode='valid')  # Convolve with original image
    C2 = np.uint8(C2) # Quantisation in uint8
    return C2, B

def noise(blurImg, var):  # To add gaussian noise to an image
	row,col = blurImg.shape
	mean = 0
	sigma = var**0.5

	gauss = np.random.normal(mean,sigma,(row,col)) # Adding gaussian distribution to an array
	gauss = np.uint8(gauss)
	noisy = blurImg + gauss # Adding noise
	return noisy, gauss

#======================= Getting the Filter ========================
def getWeiner(h,g,k):  
	H = np.fft.fft2(h, g.shape) #DFT of blur kernal
	G = np.fft.fft2(g) #DFT of noisy image

	H_conj = np.conj(H) #Conjugate of h
	H_sqr = np.multiply(H,H_conj) #|H|**2

	K = k #SNR constant

	W = np.divide(H_conj,(H_sqr + K)) 
	return W # Wiener filter

def test_rest_img(W, g, gray):
	G = np.fft.fft2(g) #DFT of noisy image
	F_rest = np.multiply(W,G) 
	f_rest = np.fft.ifft2(F_rest) #IFT of restored image
	f_rest = np.uint8(f_rest)
	f_rest = cv2.resize(f_rest, gray.shape)
	return f_rest

#============================== Metric =============================
def mse(img1,img2):
	return mean_squared_error(img1, img2) # Mean square error
 
def psnr(img1, img2): #Peak signal to noise ratio
    mse = np.mean( (img1 - img2) ** 2 )
    if mse == 0:
    	return 100
    pixel = 255.0
    return 20 * math.log10(pixel / math.sqrt(mse))

#============================ Run ===============================
# try:	
img_name = str(sys.argv[2])
kernsize = int(sys.argv[3])
var_blur = int(sys.argv[4])
var_noise = int(sys.argv[5])
try:
	k = float(sys.argv[6])
except:
	k = 0
flg = str(sys.argv[1])

image = cv2.imread(img_name) #Reading image
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #Getting Grayscale

blurImg, h = blur(kernsize,var_blur,gray) #Blurring the image and return blur kernal alongwith it
g, gauss = noise(blurImg,var_noise) #Adding noise and return noise function alongwith it


if flg == 'train': #for training on a image to get weiner

	print "\nTraining on %s\n"%(img_name)

	W  = getWeiner(h,g,k)  #Get weiner given a k value
	f_rest = test_rest_img(W,g,gray) #Restoring image through that k

	mse_ = mse(f_rest,gray) 
	print "MSE for K = %f: %f"%(k,mse_)
	psnr_ = psnr(f_rest,gray)
	print "PSNR for K = %f: %f\n"%(k,psnr_)
	cv2.imwrite('train/Rest_%f.png'%(k), f_rest)

	list_op = [kernsize, var_blur, var_noise, k]
	np.save('trained_options.npy',list_op)  #Saving filter features to apply on other images
	print "Weiner Filter saved\n"

	cv2.imwrite('train/Gray.png', gray)
	cv2.imwrite('train/Noise.png', g)

elif flg == 'test':

	print "\nTesting on %s\n"%(img_name)

	list_op = np.load('trained_options.npy') #Loading the filter features
	blurImg_t, h_t = blur(int(list_op[0]),int(list_op[1]),gray) 
	
	W  = getWeiner(h_t,g,list_op[3]) #Getting the filter using features
	print "Weiner Filter loaded\n"

	f_rest = test_rest_img(W,g,gray) #Using the filter to restore the image

	mse_ = mse(f_rest,gray)
	print "MSE: %f"%(mse_)
	psnr_ = psnr(f_rest,gray)
	print "PSNR: %f\n"%(psnr_)

	cv2.imshow('Noisy Image',g)
	cv2.imshow('Grayed Image (Clear)',gray)
	cv2.imshow('Restored (Check Console for Metric Values)',f_rest)
	
	cv2.imwrite('test/Noisy_%s'%(img_name), g)
	cv2.imwrite('test/Restored_%s'%(img_name), f_rest)

	cv2.waitKey(0) 
	cv2.destroyAllWindows() 
else: 
	j = i + o
# except:
# 	print "\nError:\n Check the input options - python weiner.py [train/test] [IMG_NAME] [BLUR_KERN_SIZE] [VAR_BLUR] [VAR_NOISE] [K]\n"
# 	print " OR \n"
# 	print "Input dimensions not a square\n"


