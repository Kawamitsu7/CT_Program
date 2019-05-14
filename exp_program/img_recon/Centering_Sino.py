#-*- coding:utf-8 -*-
import cv2
import numpy as np
import math
import os
import matplotlib.pyplot as plt
import glob

import time

import Check as CH

func_time = []

def main(f,f_name):
	print(f_name)

	global func_time
	func_time.append(time.time())
	#画像の読み込み
	#出力フォルダの作成
	os.makedirs("CENT_img", exist_ok = True)

	#中心推定および補正関数の実行
	input_f = cv2.imread(f,-1)
	L,R = center_detect(input_f)
	input_f = cv2.imread(f,-1)
	centered_f = centering(input_f,L,R)
	cv2.imwrite("CENT_img" + os.sep + "Centered.tif", centered_f.astype(np.uint16))

	func_time.append(time.time())
	return func_time[3] - func_time[2] + func_time[1] - func_time[0]

#center_detectで得た物体の端のピクセルから中心計算、画像を平行移動して出力
#入力:画像(numpy行列) 物体の端のピクセル(平均値), 出力:画像(numpy行列)
def centering(I_im,L,R):
	#rows,cols,ch = I_im.shape
	print((L,R))

	center = (R+L)/2

	#print(center)

	#画像移動用のアフィン変換に用いる、変換行列の計算
	M = np.float32([[1,0,(I_im.shape[1]/2 - center)],[0,1,0]])
	rows,cols = I_im.shape

	#画像の再配置
	#画像の端の処理 (案 : 一番外の画素値と一致させる) 
	dst = cv2.warpAffine(I_im,M,(cols,rows),borderMode = cv2.BORDER_REPLICATE)

	return dst

#断層検査物体のエッジを検出する。
#入力 : 画像(numpy行列), 注目する行の番号(int)
#出力 : 値(物体の端のピクセル)
def center_detect(I_im):
	global func_time
	func_time.append(time.time())

	#入力画像のある行に注目して画素値を検証
	#2値化で用いる閾値(ImageJでplotした結果をもとに実験値代入)
	print("Centering前の画像で、ピクセル値プロファイルを確認してください")
	THRESHOLD = input("中心推定のための2値化閾値を入力 >>")

	func_time.append(time.time())

	#閾値で2値化する。
	bi_im = I_im.copy().astype(np.int32)
	bi_im[bi_im < int(THRESHOLD)] = 0
	bi_im[bi_im >= int(THRESHOLD)] = 255

	'''
	kernel = np.ones((15,15),np.uint16)

	result = cv2.dilate(edges, kernel, iterations = 1)
	result = cv2.dilate(result, kernel, iterations = 2)
	result = cv2.erode(result, kernel, iterations = 3)

	kernel = np.ones((5,5),np.uint16)
	result = cv2.erode(result, kernel, iterations = 2)
	result = cv2.dilate(result, kernel, iterations = 2)
	'''

	l_temp = [-1]*bi_im.shape[0]
	r_temp = [-1]*bi_im.shape[0]

	#二値化画像から物体の端を検出する(1行ごとに)
	for m in range (bi_im.shape[0]):
		for j in range (bi_im.shape[1] - 1):
			if bi_im[m,j+1] - bi_im[m,j] > 200 :
				l_temp[m] = j+1
			if l_temp[m] != -1 and bi_im[m,j+1] - bi_im[m,j] < -200 :
				r_temp[m] = j
		print((str(m) + "行目",l_temp[m],r_temp[m]))

	l_ave = sum(l_temp)/len(l_temp)
	r_ave = sum(r_temp)/len(r_temp)

	return l_ave, r_ave
