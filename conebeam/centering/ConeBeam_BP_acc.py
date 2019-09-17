# -*- coding:utf-8 -*-
import cv2
import numpy as np
import os
import math
from tqdm import tqdm, trange
from numba import jit

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
	weighten_table, qu_table, qv_table = create_table(in_f.shape[1], in_f.shape[1], len(files), in_f.shape[0],sin_table, cos_table, Lo, Ld, px_size)

	u_temp = 0
	v_temp = 0

	# print(qu_table[20, 20, :])
	# print(qv_table[20, 20, :, 45])

	# デバッグ用(z軸方向中央付近で試す)
	# z_list = [45]

	#Vol_Calc(files, in_f, Volume, qu_table, qv_table, weighten_table)
	Vol_Calc_acc(files, in_f, Volume, qu_table, qv_table, weighten_table)

	Normalized = normalize * Volume
	# print("Max = " + str(np.amax(Normalized)) )
	# print("min = " + str(np.amin(Normalized)) )

	Fixed = np.where(Normalized < 0, 0, Normalized)

	Output = Cube_LI(Fixed)

	np.round(Output)

	# 画像書き出し
	for i in trange(Output.shape[2], desc='Write-img', leave=True):
		# for i in z_list:
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
def create_table(co_x, co_y, co_n, co_z, sin_arr, cos_arr, d, D, pxsize):

	a = np.arange(co_x)
	b = np.arange(co_y)
	c = np.arange(co_z)

	# デバッグ用(z軸方向中央付近で試す)
	# e_list = [45]

	x = (-1 * co_x / 2 + a) * pxsize
	y = (co_y / 2 - b) * pxsize
	z = (co_z / 2 - c) * pxsize

	x = x.reshape((x.shape[0], 1))
	y = y.reshape((y.shape[0], 1))
	#z = z.reshape((z.shape[0], 1))

	sin = sin_arr.reshape((1,sin_arr.shape[0]))
	cos = cos_arr.reshape((1,sin_arr.shape[0]))

	xcos = x * cos
	ysin = y * sin

	xcos = xcos.reshape((xcos.shape[0], 1, xcos.shape[1]))
	ysin = ysin.reshape((1, ysin.shape[0], ysin.shape[1]))

	denomi = d - xcos + ysin

	W = d / denomi
	w_table = np.square(W)

	print("weight_calculated")

	xsin = x * sin
	ycos = y * cos
	xsin = xsin.reshape((xsin.shape[0], 1, xsin.shape[1]))
	ycos = ycos.reshape((1, ycos.shape[0], ycos.shape[1]))

	u_numer = xsin + ycos
	U = u_numer / denomi

	u_ind = U * D / pxsize
	u_table = np.where(np.abs(u_ind) < co_x / 2, u_ind + co_x / 2, 2 * co_x)

	print("u_calculated")

	z = z.reshape((1, 1, 1, z.shape[0]))
	denomi_forth = denomi.reshape((denomi.shape[0],denomi.shape[1],denomi.shape[2],1))
	V = z / denomi_forth
	v_ind = V * D / pxsize
	v_table = np.where(np.abs(v_ind) < co_z / 2, (co_z / 2) - v_ind, 2 * co_z)

	print("v_calculated")

	'''
	for e in trange(co_z, desc='Create_table', leave=True):
		# for e in e_list:
		# print("table_calc" + str(e) + "行目")
		z = ( co_z / 2 - e ) * pxsize

		for a in range(co_x):
			x = ( -1 * co_x / 2 + a ) * pxsize

			for b in range(co_y):
				y = ( co_y / 2 - b ) * pxsize

				for c in range(co_n):
					sin = sin_arr[c]
					cos = cos_arr[c]

					W = d / (d - x * cos + y * sin)
					w_table[a, b, c] = np.square(W)

					U = D * (x * sin + y * cos) / (d - x * cos + y * sin)
					u_ind = U / pxsize
					if u_ind < co_x / 2 and u_ind > -1 * co_x / 2:
						u_table[a, b, c] = u_ind + co_x / 2
					else:
						u_table[a, b, c] = 2 * co_x

					V = D * z / (d - x * cos + y * sin)
					v_ind = V / pxsize
					if v_ind < co_z / 2 and v_ind > -1 * co_z / 2:
						v_table[a, b, c, e] = (co_z / 2) - v_ind
					else:
						v_table[a, b, c, e] = 2 * co_z
	'''

	return w_table, u_table, v_table


def Vol_Calc_acc(files, in_f, Volume, qu_table, qv_table, weighten_table):
	u = np.floor(qu_table).astype(int)
	v = np.floor(qv_table).astype(int)

	u_eliminate = np.where(qu_table < in_f.shape[1], 1, 0)
	v_eliminate = np.where(qv_table < in_f.shape[0], 1, 0)

	diff_u = qu_table - u
	diff_v = qv_table - v

	diff_u = np.where(u >= in_f.shape[1]-1, 1, diff_u).reshape((diff_u.shape[0], diff_u.shape[1], diff_u.shape[2], 1))
	diff_v = np.where(v >= in_f.shape[0]-1, 1, diff_v)
	u = np.where(u >= in_f.shape[1]-1, u-1, u)
	v = np.where(v >= in_f.shape[0]-1, v-1, v)

	u = u * u_eliminate
	v = v * v_eliminate

	u = u.reshape((u.shape[0], u.shape[1], u.shape[2], 1))

	print(type(files[0]))

	for n in trange(len(files), desc='Vol_Calc', leave=True):
		proj_val = (1 - diff_u[:, :, n, :]) * (1 - diff_v[:, :, n, :]) * files[n][v[:, :, n, :], u[:, :, n, :]]
		proj_val += (1 - diff_u[:, :, n, :]) * diff_v[:, :, n, :] * files[n][v[:, :, n, :] + 1, u[:, :, n, :]]
		proj_val += diff_u[:, :, n, :] * (1 - diff_v[:, :, n, :]) * files[n][v[:, :, n, :], u[:, :, n, :] + 1]
		proj_val += diff_u[:, :, n, :] * diff_v[:, :, n, :] * files[n][v[:, :, n, :] + 1, u[:, :, n, :] + 1]

		proj_val = proj_val * u_eliminate[:, :, n].reshape((u_eliminate.shape[0], u_eliminate.shape[1], 1)) * v_eliminate[:, :, n, :]

		Volume += weighten_table[:, :, n].reshape((weighten_table.shape[0], weighten_table.shape[1], 1)) * proj_val

def Vol_Calc(files, in_f, Volume, qu_table, qv_table, weighten_table):
	for z in trange(Volume.shape[2], desc='W-BP', leave=True):
		# for z in z_list:
		# print("BP : {z}行目".format(z=z))
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

def Cube_LI(Vol):
	floor = np.amin(Vol)
	ceil = np.amax(Vol)

	coefficient = 65535 / (ceil - floor)

	temp = Vol - floor

	return temp * coefficient
