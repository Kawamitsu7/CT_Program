# -*- coding:utf-8 -*-
import cv2
import numpy as np
import math
import os
import matplotlib.pyplot as plt
import glob

import linear_transform as LI
import Check as CH


# フィルタリング関数

def filtering(I_im):
	# 実験データの端に白いピクセルがあったので黒くする

	os.makedirs("fil_img", exist_ok=True)

	T_im = np.zeros((I_im.shape[0], I_im.shape[1]))

	for a in range(I_im.shape[0]):
		#print("Filtering : " + str(a) + "行目")
		for_filtering_data = np.copy(I_im[a, :])
		fft_data = np.fft.fft(for_filtering_data, I_im.shape[1])
		omega = np.fft.fftfreq(I_im.shape[1], 1)

		filtered_data = []
		for n in range(I_im.shape[1]):
			filtered_data.append(fft_data[n] * abs(omega[n]) * 2 * math.pi)

		T_im[a, :] = np.real(np.fft.ifft(filtered_data, I_im.shape[1]))

	O_im = np.zeros((I_im.shape[0], I_im.shape[1]))
	# LI.linear_transform_calc(T_im, O_im, np.amin(T_im), np.amax(T_im))

	# print(omega)
	# print(np.amax(fft_data))
	# print(np.amin(fft_data))
	# CH.check_and_quit(T_im[m],1)
	# print(I_im[m])
	# quit()

	# cv2.imwrite("fil_img" + os.sep + "Filtered_img.tif", O_im.astype(np.uint16))
	O_im = T_im
	# print(O_im.dtype)
	return O_im

# quit()
