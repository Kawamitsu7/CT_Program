import cv2
import numpy as np
import math
import os
import matplotlib.pyplot as plt
import glob

import Check as CH

#線形変換に使用する一連の関数

def linear_transform(I_im,O_im):
	flg = 1
	CH.check_and_quit(I_im,0)
	peak_check(I_im)
	
	O_im.append(np.zeros((I_im.shape[0], I_im.shape[1])))
	
	while flg == 1:
		print("線形変換を実行します")
		print("階調値の最小値を入力してください")
		floor = int(input(">>> "))
		print("階調値の最大値を入力してください")
		ceil = int(input(">>> "))

		linear_transform_calc(I_im,O_im,floor,ceil)

		peak_check(O_im.astype(np.uint16))
		
		cv2.imshow("Plz press a key",O_im.astype(np.uint16))
		cv2.waitKey(0)
		cv2.destroyAllWindows()
		
		print("再度変換をやり直しますか : yes = 1, no = 0")
		flg = int(input(">>> "))
		if flg == 0 :
			for l in range(len(I_im) - 1):
				O_im.append(np.zeros((I_im.shape[0], I_im.shape[1])))
				linear_transform_calc(I_im,O_im,floor,ceil)
				
				print(str(l) + "step")
			return 0
	
def linear_transform_calc(I_im,O_im,floor,ceil):
	for x in range(I_im.shape[0]):
		for y in range(I_im.shape[1]):
			if I_im[x,y] < floor :
				temp = 0
			else :
				temp = I_im[x,y] - floor
			
			O_im[x,y] = round(65535 /(ceil - floor) * temp)

def peak_check(I_im):
	check_peak_array = [0]*65536

	for x in range(I_im.shape[0]):
		for y in range(I_im.shape[1]):
			check_peak_array[I_im[x,y]] += 1

	plt.plot(check_peak_array)
	plt.show()	

