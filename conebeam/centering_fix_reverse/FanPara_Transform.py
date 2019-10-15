#-*- coding:utf-8 -*-
import cv2
import numpy as np
import math
import os
import matplotlib.pyplot as plt
import glob

def main(f,num):

	Lo = 8/100
	Ld = 12/100

	input_f = f

	px_size = 83 / 1000000 * (1280 / input_f.shape[1])

	os.makedirs("FP_Trans", exist_ok = True)
	para_im = F_to_P_Trans(input_f,Lo,Ld,px_size)
	
	cv2.imwrite("FP_Trans"+ os.sep +"F2P_"+ str(num) +".tif", para_im.astype(np.uint16))

	return para_im
	
def F_to_P_Trans(I_im,Lo,Ld,px_size):
	dst = np.zeros((I_im.shape[0],I_im.shape[1]))
	delta_theta = 360 / I_im.shape[0]

	for theta_p in range (I_im.shape[0]):
		# print("FP_trans : " + str(theta_p + 1) + "行目")
	
		for Xp in range (I_im.shape[1]):
			Xf = Calc_Xf(Xp, px_size, Lo, Ld, I_im.shape[1])
			theta_f = Calc_theta_f(theta_p, px_size, Xp, Lo, I_im.shape[1], delta_theta)
			dst[theta_p,Xp] = interpolation(I_im, theta_f,Xf)

	return dst

def Calc_Xf(Xp, px_size, Lo, Ld, array_num):
	real_Xp = (Xp - array_num/2 - 1/2) * px_size
	real_Xf = Ld * real_Xp / math.sqrt(Lo ** 2 - real_Xp ** 2)
	return real_Xf / px_size + array_num/2 + 1/2

def Calc_theta_f(theta_p, px_size, Xp, Lo, array_num, delta_theta):
	real_Xp = (Xp - array_num/2 - 1/2) * px_size
	alpha = np.arctan( real_Xp / math.sqrt(Lo ** 2 - real_Xp ** 2) )
	return ( theta_p * delta_theta - np.rad2deg(alpha) ) / delta_theta

def interpolation(I_im, theta_f, Xf):
	x,y = fold_back(theta_f,Xf,I_im.shape)

	x = math.floor(x)
	y = math.floor(y)
	if Xf < 0:
		diff_x = 0
	elif Xf > I_im.shape[1]:
		diff_x = 1
	else:
		diff_x = Xf - x
	
	if theta_f < 0:
		diff_y = 0
	elif theta_f > I_im.shape[0]:
		diff_y = 1
	else:
		diff_y = theta_f - y

	dst = cv2.copyMakeBorder(I_im, 0, 1, 0, 1, cv2.BORDER_REPLICATE)

	left_up = dst[y,x] * (1-diff_x) * (1-diff_y)
	left_down = dst[y+1,x] * (1-diff_x) * diff_y
	right_up = dst[y,x+1] * diff_x * (1-diff_y)
	right_down = dst[y+1,x+1] * diff_x * diff_y

	return left_up + left_down + right_up + right_down

def fold_back(y,x,img_size):
	if x >= img_size[1]:
		#print("over flow x : "+str(x))
		x = img_size[1] - 1
		
		if x >= img_size[1]:
			print("x error :" + str(x))
	elif x < 0:
		#print("under flow x : "+str(x))
		x = 0

	if y >= img_size[0]:
		#print("over flow y : "+str(y))
		y = y - img_size[0]
	elif y < 0:
		#print("under flow y : "+str(y))
		y = img_size[0] + y

	return x,y