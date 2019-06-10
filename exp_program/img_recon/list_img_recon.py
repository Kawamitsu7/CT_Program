#-*- coding:utf-8 -*-
import cv2
import numpy as np
import math
import os
import matplotlib.pyplot as plt
import glob

import time

import FanPara_Transform as F2P
import Centering_Sino as CenS
import FBP_Sinogram as FBP

# -*- All in One program -*-
# src_img(Sinogram) -> FanPara_Trans -> Centering_Sino -> FBP(Filtering -> BP) -> dst_img(reconstructed tomography)

def main(f,F2P_flg,CentS_flg,log,flgCentS,center):
	start_time = 0
	fin_time = 0
	func_time = 0
	sum_time = 0

	#ファイル名の調整
	f_name, _ = os.path.splitext( os.path.basename(f) )
	
	#Fan -> Para 変換
	start_time = time.time()
	if F2P_flg == 1:
		F2P.main(f)
		inter_f = "FP_Trans"+ os.sep +"F_to_P.tif"
	else :
		inter_f = f
	fin_time = time.time()

	print("FanPara_Transform.py : {}sec".format(fin_time - start_time), file=log)
	sum_time += fin_time - start_time

	#Centering_Sino
	if CentS_flg == 1:
		func_time = CenS.main(inter_f,f_name,flgCentS,center)
		inter_f = "CENT_img" + os.sep + "Centered.tif"

	print("Centering_Sino.py : {}sec".format(func_time), file=log)
	sum_time += func_time
	

	#FBP
	start_time = time.time()
	FBP.main(inter_f,f_name)
	fin_time = time.time()

	print("FBP_sinogram.py : {}sec".format(fin_time - start_time), file=log)
	sum_time += fin_time - start_time

	print("Time of All function : {}sec".format(sum_time), file = log)

if __name__ == "__main__":
	#実行時間ログの作成
	log = open('timelog.txt','w')
	print("~Image Reconstruction : time log~", file=log)
	
	#画像の読み込み
	folder = input("投影データ(サイノグラム)のディレクトリを選択してください >>>")
	files = glob.glob(folder + os.sep + "*.tif")
	files.sort()

	#各種前処理を行うかどうかの選択
	isF2P = int(input("ファンパラ変換を行うか選択 (0:実行しない 1:実行する) >>>"))
	isCentS = int(input("サイノグラム中心補正を行うか選択 (0:実行しない 1:実行する) >>>"))

	flgCentS = 0
	center = 0.0
	if isCentS == 1:
		flgCentS = int(input("中心補正:1枚ずつ-->0 まとめてやる-->1 >>>"))
		center = float(input("まとめて中心の値指定 >>>"))

	for f in files:
		print("-------------\n ==="+ os.path.basename(f) +"===", file=log)
		main(f,isF2P,isCentS,log,flgCentS,center)

	log.close()
