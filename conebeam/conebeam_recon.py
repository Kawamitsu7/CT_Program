# -*- coding:utf-8 -*-
import cv2
import numpy as np
import os
import glob
import time

import proj_mask as PM
import Filtering_Proj as Fil_P
import linear_transform as LI
import ConeBeam_BP as CBBP

# -*- All in One program -*-
# src_img -> masking -> Filtering -> weighted-BP -> dst_img

if __name__ == "__main__":

	# 実行時間ログの作成
	log = open('timelog.txt', 'w')
	print("~Conebeam Reconstruction : time log~", file=log)

	# 画像の読み込み
	folder = input("投影データのディレクトリを選択してください >>>")
	files = glob.glob(folder + os.sep + "*.tif")
	files.sort()

	# 各種前処理を行うかどうかの選択
	# isCentS = int(input("サイノグラム中心補正を行うか選択 (0:実行しない 1:実行する) >>>"))

	# flgCentS = 0
	# center = 0.0
	# if isCentS == 1:
	# 	flgCentS = int(input("中心補正:1枚ずつ-->0 まとめてやる-->1 >>>"))
	# 	center = float(input("まとめて中心の値指定 >>>"))

	print("-----------", file=log)
	fil_arr = []

	start_time = time.time()
	iter_count = 0

	for f in files:
		# ファイル名の調整
		f_name, _ = os.path.splitext(os.path.basename(f))

		# masking
		masked = PM.main(f)

		# filtering
		O_im = np.zeros((masked.shape[0], masked.shape[1]))
		filtered = Fil_P.filtering(masked)
		LI.linear_transform_calc(filtered, O_im, np.amin(filtered), np.amax(filtered))
		cv2.imwrite("fil_img" + os.sep + "Filtered_" + f_name + ".tif", O_im.astype(np.uint16))

		fil_arr.append(filtered)
		iter_count += 1
		print(str(iter_count) + "枚目 Filtered")

	end_time = time.time()
	print("project_masking & filtering : {}sec".format(end_time - start_time), file=log)

	# 逆投影パート
	print("-----------", file=log)
	start_time = time.time()

	CBBP.Cone_BP(fil_arr)

	end_time = time.time()
	print("Weighten Back Projection : {}sec".format(end_time - start_time), file=log)

	log.close()
