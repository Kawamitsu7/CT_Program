#-*- coding:utf-8 -*-
import cv2
import numpy as np
import math
import os
import matplotlib.pyplot as plt
import glob
import time

def main(src,s,f,isGray):
	#画像の読み込み
	folder = src

	print("画像書き出しを開始するスライスを指定してください(0~画像サイズ-1)")
	start = int(s)
	print("画像書き出しを終了するスライスを指定してください(0~画像サイズ-1)")
	fin = int(f)
	print(str(start) + "枚目から" + str(fin) + "枚目までを出力します")

	print("グレースケール反転の有無を選択してください(1:反転する/0:反転しない)")
	gray_flg = int(isGray)

	input_im = []
	files = glob.glob(folder + os.sep + "*.tif")
	files.sort()
	#print(files)

	for k in files:
		input_f = cv2.imread(k,-1)
		print(str(k))
		if gray_flg == 1:
			input_f = cv2.bitwise_not(input_f)
		input_im.append(input_f[start:(fin+1),:]) #8/6 変更　imreadで転置済み二次元配列に

	if len(input_im) == 0:
		print("Failed!")
	else:
		print("Success! : Read Image")
	
	os.makedirs("Sinogram", exist_ok = True)
	sino_arr = []
	for j in range (fin + 1 - start):
		sino = Create_Sinogram(input_im,j,start)
		cv2.imwrite("Sinogram"+ os.sep + str(src) + "_Sino_" + str(start + j).zfill(4) + ".tif", sino.astype(np.uint16))
		sino_arr.append(sino)

	return sino_arr
	
def Create_Sinogram(I_im,j,start):
	dst = np.zeros((len(I_im),I_im[j].shape[1]))
	for i in range (len(I_im)):
		dst[i,:] = I_im[i][j,:]

	return dst
		
if __name__ == "__main__":
	start_time = 0
	process_time = 0
	log = open('timelog.txt','w')
	print("~Create_Sinogram : time log~", file=log)

	for line in open("list.txt","r",encoding="utf-8"):
		if line[0] == "#" or line[0] == "\n":
			continue
		data = line.split('\t')

		src = data[0]
		s = data[1]
		f = data[2]
		isGray = data[3].strip()

		#print("{}{}{}{}".format(src,s,f,isGray))

		start_time = time.time()
		main(src,s,f,isGray)
		process_time = time.time() - start_time

		print(str(src) + " : {}sec".format(process_time), file=log)

	log.close()
