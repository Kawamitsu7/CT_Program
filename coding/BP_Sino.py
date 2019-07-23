#-*- coding:utf-8 -*-
import cv2
import numpy as np
import math
import os
import matplotlib.pyplot as plt
import glob

import linear_transform as LI
import Check as CH
import Filtering_Sino as FL

def back_projection(I_im,f_name):
	j = 0	#現画像の列番号(x座標に対応)
	m = 0	#回転サンプルを適応するループ用変数
	
	j_temp = 0 #jを決定するための前段階変数 : 小数
	p = 0 #p(r,theta)の値 : 補間済み小数

	cv2.imwrite("Check_img.tif",I_im.astype(np.uint16))
	
	#逆投影時に参照する投影データを決定するための行列を導出
	j_matrix = np.zeros((I_im.shape[1], I_im.shape[1],I_im.shape[0]))	
	calc_BP_matrix(j_matrix)
			
	#逆投影画像の初期化
	to_imaging = np.zeros((I_im.shape[1], I_im.shape[1]))
	for a in range(I_im.shape[1]):
		print("BP : {a}行目".format(a = a))

		for b in range(I_im.shape[1]):
			#print("BP : ({a},{b})".format(a = a,b = b))
			for m in range(I_im.shape[0]):
				if j_matrix[a,b,m] < I_im.shape[1]:
					j_temp = j_matrix[a,b,m]
				#線形補間
				j = math.floor(j_temp)
				if j >= I_im.shape[1] - 1:
					p = I_im[m,I_im.shape[1] - 1]
				else :
					p = (j_temp - j) * I_im[m,j] + (j + 1 - j_temp) * I_im[m,j+1]
				to_imaging[a,b] += p
				#to_imaging[(to_imaging.shape[0] - 1) - a, (to_imaging.shape[1] - 1) - b] += p	#180°(半周分)しか投影データがない為、残り半周分を対象データとして計算する
		#quit()
	
	
		#画像の補正(要検討)
	fixed = fix(to_imaging,I_im.shape[0])

	print(np.amax(fixed))
	print(np.amin(fixed))

	output_img = np.zeros((I_im.shape[1], I_im.shape[1]))

	output_img = np.where(fixed < 0, 0, fixed)

	LI.linear_transform_calc(fixed,output_img,np.amin(output_img),np.amax(output_img))

	np.round(output_img)
	
	#画像書き出し
	cv2.imwrite("BP_data" + os.sep + str(f_name) +"_BP.tif", output_img.astype(np.uint16))

'''
		#投影画像を作成(変更する)
		BP_im = []
		im_num = 0
		while im_num < 30:
			temp = np.repeat(I_im[im_num][i,:], I_im[0].shape[1], axis = 1)	#I_im[im_num]の(i+1)行目の値を引き延ばした画像作成
			BP_im.append(temp.reshape(I_im[0].shape[0],I_im[0].shape[1],3))
			im_num += 1

		#投影画像の回転(変更する)
		rotate_im = []
		im_num = 0
		while im_num < 30:
			R = cv2.getRotationMatrix2D((ox, oy), theta * im_num, 1.0)			#回転変換行列の導出
			src = BP_im[im_num]
			dsize = src.shape[0:2]	#画像サイズなので要素は最初の2個だけ

			temp = cv2.warpAffine(src, R, dsize, flags=cv2.INTER_CUBIC)
			rotate_im.append(temp)	#現画像をアフィン変換して代入
			im_num += 1
			
		#回転画像の足し算で逆投影(変更する)
		im_num = 0
		to_imaging = np.zeros((I_im[0].shape[1], I_im[0].shape[1], 3))
		while im_num < 30:
			to_imaging += rotate_im[im_num]		
			im_num += 1
'''

def calc_BP_matrix(matrix):
	a = 0	#作成画像の画素指定ループ用変数(x座標用,資料でいうhのこと)
	b = 0	#作成画像の画素指定ループ用変数(y座標用,資料でいうvのこと)
	delta = 1 #画素値の座標空間での間隔(delta = 2とするほうが楽かも…)

	m = 0	#回転サンプルを適応するループ用変数
	theta = 360 / matrix.shape[2]	#回転サンプル間の回転角(画像セットに応じて適宜変更)
	
	r = 0	#角度(theta*m)に対して、(x,y)の投影データが中心からどれだけ離れているか

	for a in range(matrix.shape[0]):
		print("BP_Mat_Calc : {a}行目".format(a = a))

		for b in range(matrix.shape[1]):
			#print("BP_Mat_Calc : ({a},{b})".format(a = a,b = b))

			x = delta * (-1 * matrix.shape[0] / 2 + a)			
			y = delta * (matrix.shape[1] / 2 - b)
			
			for m in range(matrix.shape[2]):
				#print("BP_mat_Calc : " + str(m) + "step")

				r = x * math.cos(math.radians(theta * m)) + y * math.sin(math.radians(theta * m))
				if r < matrix.shape[1] / 2 and r > -1 * matrix.shape[1] / 2:
					matrix[a,b,m] = r / delta + matrix.shape[1] / 2
				else :
					matrix[a,b,m] = 2 * matrix.shape[1]
			#print("(a,b)=",a,b)


#def lerp(i)	
		
def fix(im, m):

	result = np.zeros((im.shape[0], im.shape[1]))

	#逆投影の値を投影数で除算し、画素値にする
	#画素値は離散整数なので、round関数を使用
	for a in range(im.shape[0]):
		for b in range(im.shape[1]):
			result[a,b] = im[a,b] / (m * 2)

	return result
