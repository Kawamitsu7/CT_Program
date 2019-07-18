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
import BP_Sino as BP

def main(f,f_name):
	#print(f_name)
	#quit()
	#画像の読み込み
	input_f = cv2.imread(f,-1)

	'''
	#事前に入力画像の優位なスケーリングを行う
	#線形変換を採用
	transformed_im = []	#2次元配列(画素値)を格納する配列,要素は30
	linear_transform(input_im, transformed_im)
	'''

	#フィルタリング
	filtered_im = np.zeros((input_f.shape[0],input_f.shape[1]))	#2次元配列(画素値)を格納する配列,要素は30
	filtered_im = FL.filtering(input_f)
	
	#Back-Projection(画像は指定枚数生成される)
	#2次元配列(画素値)を格納する,要素は512*512
	os.makedirs("BP_data", exist_ok = True)
	BP.back_projection(filtered_im,f_name)	#画像補正はこの関数内のfix()で実施
