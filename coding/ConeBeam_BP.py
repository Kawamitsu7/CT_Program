# -*- coding:utf-8 -*-
import cv2
import numpy as np
import os
import glob
import time
import math
from joblib import Parallel, delayed
from numba import jit
from tqdm import tqdm, trange

def Cone_BP(files):
	# ----------逆投影に必要なパラメータ準備----------

	# 距離 : 線源 --> 回転中心
	Lo = 8 / 100

	# 距離 : 線源 --> 検出器
	Ld = 12 / 100

	in_f = files[0]

	# 検出器1個当たりの大きさ
	px_size = 83 / 1000000 * (1280 / in_f.shape[1])

	os.makedirs("CB_BP", exist_ok=True)

	Volume = np.zeros((in_f.shape[1], in_f.shape[1], in_f.shape[0]))

	d_theta = 360.0 / len(files)

	# 再構成のラストにかける正規化係数
	# normalize = 1.0 / (2 * np.pi * len(files))
	normalize = 1.0 / (2 * len(files))

	sin_table, cos_table = sincos_calc(len(files), d_theta)

	# ----------ここまで----------

	# ==========再構成処理本体==========

	# 再構成テーブルを作成
	weighten_table, qu_table, qv_table = create_table(in_f.shape[1], in_f.shape[1], len(files), in_f.shape[0],
	                                                  sin_table, cos_table, Lo, Ld)

	u_temp = 0
	v_temp = 0

	# デバッグ用(z軸方向中央付近で試す)
	z_list = [45]

	# for z in range(Volume.shape[2]):
	for z in z_list:
		print("BP : {z}行目".format(z=z))
		for x in range(Volume.shape[0]):
			for y in range(Volume.shape[1]):
				for n in range(len(files)):
					proj_val = 0.0
					diff_u = 0.0
					diff_v = 0.0

					# 双線形補間
					if qu_table[x, y, n] < in_f.shape[1] and qv_table[x, y, n, z] < in_f.shape[0]:
						u_temp = qu_table[x, y, n]
						v_temp = qv_table[x, y, n, z]

						u = math.floor(u_temp)
						v = math.floor(v_temp)
						if u >= in_f.shape[1] - 1 and v >= in_f.shape[0] - 1:
							proj_val = files[n][in_f.shape[0] - 1, in_f.shape[1] - 1]
						elif u >= in_f.shape[1] - 1:
							diff_v = v_temp - v
							proj_val = (1 - diff_v) * files[n][v, in_f.shape[1] - 1] + diff_v * files[n][
								v + 1, in_f.shape[1] - 1]
						elif v >= in_f.shape[0] - 1:
							diff_u = u_temp - u
							proj_val = (1 - diff_u) * files[n][in_f.shape[0] - 1, u] + diff_u * files[n][
								in_f.shape[0] - 1, u + 1]
						else:
							diff_u = u_temp - u
							diff_v = v_temp - v
							left_up = (1 - diff_u) * (1 - diff_v) * files[n][v, u]
							left_down = (1 - diff_u) * diff_v * files[n][v + 1, u]
							right_up = diff_u * (1 - diff_v) * files[n][v, u + 1]
							right_down = diff_u * diff_v * files[n][v + 1, u + 1]
							proj_val = left_up + left_down + right_up + right_down

					Volume[x, y, z] += weighten_table[x, y, n] * proj_val

	Normalized = normalize * Volume

	Fixed = np.where(Normalized < 0, 0, Normalized)

	Output = Cube_LI(Fixed)

	np.round(Output)

	# 画像書き出し
	# for i in range(Output.shape[2]):
	for i in z_list:
		to_img = Output[:, :, i]
		cv2.imwrite("CB_BP" + os.sep + str(i) + "_BP.tif", to_img.astype(np.uint16))

	print("End of Reconstruction")


# 三角関数の計算
def sincos_calc(n, d_theta):
	sin_arr = np.zeros(n)
	cos_arr = np.zeros(n)
	theta = 0.0

	for i in range(n):
		rad = np.deg2rad(theta)
		sin_arr[i] = np.sin(rad)
		cos_arr[i] = np.cos(rad)
		theta += d_theta

	return sin_arr, cos_arr


# 重みづけ関数
# z座標に関係ないので先にテーブル作っとくべき？
def create_table(co_x, co_y, co_n, co_z, sin_arr, cos_arr, d, D):
	w_table = np.zeros((co_x, co_y, co_n))
	u_table = np.zeros((co_x, co_y, co_n))
	v_table = np.zeros((co_x, co_y, co_n, co_z))

	# デバッグ用(z軸方向中央付近で試す)
	e_list = [45]

	# for e in range(co_z):
	for e in e_list:
		print("table_calc" + str(e) + "行目")
		z = co_z / 2 - e

		for a in range(co_x):
			x = -1 * co_x / 2 + a

			for b in range(co_y):
				y = co_y / 2 - b

				for c in range(co_n):
					sin = sin_arr[c]
					cos = cos_arr[c]

					W = d / (d - x * cos + y * sin)
					w_table[a, b, c] = np.square(W)

					U = D * (x * sin + y * cos) / (d - x * cos + y * sin)
					if U < co_x / 2 and U > -1 * co_x / 2:
						u_table[a, b, c] = U + co_x / 2
					else:
						u_table[a, b, c] = 2 * co_x

					V = D * z / (d - x * cos + y * sin)
					if V < co_z / 2 and V > -1 * co_z / 2:
						v_table[a, b, c, e] = (co_z / 2) - V
					else:
						v_table[a, b, c, e] = 2 * co_z

	return w_table, u_table, v_table


def Cube_LI(Vol):
	floor = np.amin(Vol)
	ceil = np.amax(Vol)

	coefficient = 65535 / (ceil - floor)

	temp = Vol - floor

	return temp * coefficient
