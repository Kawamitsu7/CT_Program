# -*- coding:utf-8 -*-
import cv2
import numpy as np
import os
import glob
import time


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
	normalize = 1.0 / (2 * np.pi * len(files))

	sin_table, cos_table = sincos_calc(len(files),d_theta)

	# ----------ここまで----------

	# ==========再構成処理本体==========

	# 再構成テーブルを作成
	weighten_table, qu_table, qv_table = create_table(in_f.shape[1], in_f.shape[1], len(files), in_f.shape[0], sin_table, cos_table, Lo, Ld)

	for f in files:
		proj = cv2.imread(f, -1)

		for x in range(Volume.shape[0]):
			for y in range(Volume.shape[0]):
				pass


	print("End of Reconstruction")


# 三角関数の計算
def sincos_calc(n,d_theta):
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
def create_table(co_x,co_y,co_n,co_z,sin_arr,cos_arr,d,D):
	w_table = np.zeros((co_x, co_y, co_n))
	u_table = np.zeros((co_x, co_y, co_n))
	v_table = np.zeros((co_x, co_y, co_n, co_z))

	for a in range(co_x):
		print(str(a) + "行目のyとz計算")
		x = -1 * co_x / 2 + a

		for b in range(co_y):
			y = co_y / 2 - b

			for c in range(co_n):
				sin = sin_arr[c]
				cos = cos_arr[c]

				W = d / (d - x * cos + y * sin)
				w_table[a,b,c] = np.square(W)

				U = D * (x * sin + y * cos) / (d - x * cos + y * sin)
				if U < co_x / 2 and U > -1 * co_x / 2:
					u_table[a, b, c] = U + co_x / 2
				else:
					u_table[a, b, c] = 2 * co_x

				for e in range(co_z):
					z = co_z / 2 - e
					V = D * z / (d - x * cos + y * sin)
					if V < co_z / 2 and V > -1 * co_z / 2:
						v_table[a, b, c, e] = (co_z / 2) - V
					else:
						v_table[a, b, c, e] = 2 * co_z


	return w_table,u_table,v_table

