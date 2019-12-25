# -*- coding:utf-8 -*-
import cv2
import numpy as np
import os
import glob
import time
from tqdm import tqdm, trange

import proj_mask as PM
import proj_mask_acc as PM_acc
import Filtering_Proj as Fil_P
import linear_transform as LI
import ConeBeam_BP_acc as CBBP

import list_sinogram as LS
import FanPara_Transform as F2P
import Centering_Sino as CS
import ParaFan_Trans as P2F

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

	# サイノグラム作成に必要な変数の設定
	in_sample = cv2.imread(files[0])
	start = 0
	fin = in_sample.shape[0] - 1

	isGray = input("グレースケール反転の有無を選択してください(1:反転する/0:反転しない) << ")

	# <再構成前処理 : 中心合わせ>
	# 投影画像->サイノグラム
	temp1 = LS.main(folder, start, fin, isGray)

	# ファンパラ変換
	temp2 = []
	for i in trange(len(temp1), desc='Fan-Para Trans', leave=True):
		temp2.append(F2P.main(temp1[i],i))

	# パラレルビーム投影の中心合わせ
	center = CS.main(temp2[int(len(temp2)/2 - 1)], 0, 0, 0)
	for i in trange(len(temp2), desc='Centering', leave=True):
		temp1[i] = CS.main(temp2[i], 1, center, i)

	# パラファン変換
	for i in trange(len(temp1), desc='Para-Fan Trans', leave=True):
		temp2[i] = P2F.main(temp1[i], i)

	# サイノグラム->投影画像
	os.makedirs("CENT_proj", exist_ok=True)
	Centered_proj = np.zeros((temp2[0].shape[0], len(temp2), temp2[0].shape[1]))
	for i in trange(len(temp2), desc='Sino -> Proj', leave=True):
		Centered_proj[:, i, :] = temp2[i]

	for i in trange(Centered_proj.shape[0], desc='CENT_Proj_Check', leave=True):
		cv2.imwrite("CENT_proj" + os.sep + "CentProj_" + str(i) + ".tif", Centered_proj[i,:,:].astype(np.uint16))

	i = 0
	for f in tqdm(files, desc='Filtering', leave=True):

		# ファイル名の調整
		f_name, _ = os.path.splitext(os.path.basename(f))

		# masking
		masked = PM_acc.main(f, Centered_proj[i, :, :])

		# filtering
		O_im = np.zeros((masked.shape[0], masked.shape[1]))
		# filtered = Fil_P.filtering(masked)
		filtered = Fil_P.fil_acc(masked)
		# LI.linear_transform_calc(filtered, O_im, np.amin(filtered), np.amax(filtered))
		O_im = LI.li_trans_calc_acc(filtered, np.amin(filtered), np.amax(filtered))
		cv2.imwrite("fil_img" + os.sep + "Filtered_" + f_name + ".tif", O_im.astype(np.uint16))

		fil_arr.append(filtered)
		# iter_count += 1
		# print(str(iter_count) + "枚目 Filtered")

		i += 1

	# print("filtered min = {}".format(np.amin(filtered)))
	# print("filtered max = {}".format(np.amax(filtered)))

	end_time = time.time()
	print("project_masking & filtering : {}sec".format(end_time - start_time), file=log)

	# quit()

	# 逆投影パート
	print("-----------", file=log)
	start_time = time.time()

	CBBP.Cone_BP(fil_arr)

	end_time = time.time()
	print("Weighten Back Projection : {}sec".format(end_time - start_time), file=log)

	log.close()
