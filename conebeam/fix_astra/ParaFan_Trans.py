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

	os.makedirs("PF_Trans", exist_ok = True)
	fan_im = P2F_acc(input_f,Lo,Ld,px_size)
	
	cv2.imwrite("PF_Trans"+ os.sep +"P2F_" + str(num) + ".tif", fan_im.astype(np.uint16))
	return fan_im

def P2F_acc(I_im,Lo,Ld,px_size):
	delta_theta = 360 / I_im.shape[0]
	theta_f = np.arange(I_im.shape[0]).reshape(I_im.shape[0],1)
	Xf = np.arange(I_im.shape[1]).reshape(1,I_im.shape[1])

	# theta_fの計算
	real_Xf = (Xf - I_im.shape[1] / 2 - 1 / 2) * px_size
	alpha = np.arctan( real_Xf / Ld )
	theta_p = theta_f + np.rad2deg(alpha)

	# Xfの計算
	real_Xp = Lo * real_Xf / np.sqrt(real_Xf ** 2 + Ld ** 2)
	Xp = real_Xp / px_size + I_im.shape[1]/2 + 1/2

	# fold_back
	theta_p = np.where(theta_p < 0, I_im.shape[0] + theta_p, theta_p)
	theta_p = np.where(theta_p >= I_im.shape[0], theta_p - I_im.shape[0], theta_p)
	Xp = np.where(Xp < 0, 0, Xp)
	Xp = np.where(Xp >= I_im.shape[1], I_im.shape[1] - 1,Xp)

	# 線形補完
	y = np.floor(theta_p).astype(int)
	x = np.floor(Xp).astype(int)
	diff_y = theta_p - y
	diff_x = Xp - x

	dst = cv2.copyMakeBorder(I_im, 0, 1, 0, 1, cv2.BORDER_REPLICATE)
	val = (1-diff_x) * (1-diff_y) * dst[y, x]
	val += (1-diff_x) * diff_y * dst[y+1, x]
	val += diff_x * (1-diff_y) * dst[y, x+1]
	val += diff_x * diff_y * dst[y+1, x+1]

	return val

def P_to_F_Trans(I_im,Lo,Ld,px_size):
	dst = np.zeros((I_im.shape[0],I_im.shape[1]))

	for theta_f in range (I_im.shape[0]):
		# print("PF_trans : " + str(theta_f + 1) + "行目")
	
		for Xf in range (I_im.shape[1]):
			Xp = Calc_Xp(Xf, px_size, Lo, Ld, I_im.shape[1])
			theta_p = Calc_theta_p(theta_f, px_size, Xf, Ld, I_im.shape[1])
			dst[theta_f,Xf] = interpolation(I_im, theta_p,Xp)

	return dst

def Calc_Xp(Xf, px_size, Lo, Ld, array_num):
	real_Xf = (Xf - array_num/2 - 1/2) * px_size
	real_Xp = Lo * real_Xf / math.sqrt(real_Xf ** 2 + Ld ** 2)
	return real_Xp / px_size + array_num/2 + 1/2

def Calc_theta_p(theta_f, px_size, Xf, Ld, array_num):
	real_Xf = (Xf - array_num/2 - 1/2) * px_size
	alpha = np.arctan( real_Xf / Ld )
	return theta_f + np.rad2deg(alpha)

def interpolation(I_im, theta_p, Xp):
	x,y = fold_back(theta_p,Xp,I_im.shape)

	x = math.floor(x)
	y = math.floor(y)
	if Xp < 0:
		diff_x = 0
	elif Xp > I_im.shape[1]:
		diff_x = 1
	else:
		diff_x = Xp - x
	
	if theta_p < 0:
		diff_y = 0
	elif theta_p > I_im.shape[0]:
		diff_y = 1
	else:
		diff_y = theta_p - y
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
		
if __name__ == "__main__":
    main()