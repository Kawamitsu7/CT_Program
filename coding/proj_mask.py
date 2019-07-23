# -*- coding:utf-8 -*-
import cv2
import numpy as np
import math
import os
import matplotlib.pyplot as plt
import glob


def main(f):
	# 距離 : 線源 --> 回転中心
	Lo = 8 / 100

	# 距離 : 線源 --> 検出器
	Ld = 12 / 100

	input_f = cv2.bitwise_not(cv2.imread(f, -1))

	# 検出器1個当たりの大きさ
	px_size = 83 / 1000000 * (1280 / input_f.shape[1])

	os.makedirs("Proj_masked", exist_ok=True)
	mask_im = Masking(input_f, Lo, Ld, px_size)

	#ファイル名の調整
	f_name, _ = os.path.splitext( os.path.basename(f) )

	cv2.imwrite("Proj_masked" + os.sep + str(f_name) +".tif", mask_im.astype(np.uint16))

	return mask_im


def Masking(I_im, Lo, Ld, px_size):
	dst = np.zeros((I_im.shape[0], I_im.shape[1]))

	for Zp in range(I_im.shape[0]):
		Z = Calc_coordinate(Zp, px_size, I_im.shape[0])

		for Xp in range(I_im.shape[1]):
			X = Calc_coordinate(Xp, px_size, I_im.shape[1])

			# 重み計算
			# Loの代わりにLdを使うべき？
			dst[Zp, Xp] = 1.0 * I_im[Zp, Xp] * Lo / math.sqrt(Lo*Lo + X*X + Z*Z)

	return dst


def Calc_coordinate(Xp, px_size, array_num):
	real_Xp = (Xp - (1.0 * array_num)/2 + 1.0/2) * px_size
	return real_Xp


# 双線形補間
def interpolation(I_im, theta_f, Xf):
	x, y = fold_back(theta_f, Xf, I_im.shape)

	x = math.floor(x)
	y = math.floor(y)
	if Xf < 0:
		diff_x = 0
	elif Xf > I_im.shape[1]:
		diff_x = 1
	else:
		diff_x = Xf - x

	if theta_f < 0:
		diff_y = 0
	elif theta_f > I_im.shape[0]:
		diff_y = 1
	else:
		diff_y = theta_f - y

	dst = cv2.copyMakeBorder(I_im, 0, 1, 0, 1, cv2.BORDER_REPLICATE)

	left_up = dst[y, x] * (1 - diff_x) * (1 - diff_y)
	left_down = dst[y + 1, x] * (1 - diff_x) * diff_y
	right_up = dst[y, x + 1] * diff_x * (1 - diff_y)
	right_down = dst[y + 1, x + 1] * diff_x * diff_y

	return left_up + left_down + right_up + right_down


# 画面端の処理
def fold_back(y, x, img_size):
	if x >= img_size[1]:
		# print("over flow x : "+str(x))
		x = img_size[1] - 1

		if x >= img_size[1]:
			print("x error :" + str(x))
	elif x < 0:
		# print("under flow x : "+str(x))
		x = 0

	if y >= img_size[0]:
		# print("over flow y : "+str(y))
		y = img_size[0] - 1
	elif y < 0:
		# print("under flow y : "+str(y))
		y = 0

	return x, y


if __name__ == "__main__":

	# 画像の読み込み
	folder = input("投影データ(サイノグラム)のディレクトリを選択してください >>>")
	files = glob.glob(folder + os.sep + "*.tif")
	files.sort()

	for f in files:
		main(f)
