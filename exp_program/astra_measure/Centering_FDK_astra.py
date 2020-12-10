# リファレンスサイト "https://tomroelandts.com/articles/astra-toolbox-tutorial-reconstruction-from-projection-images-part-2"

import numpy as np
import os
import glob
from os.path import join
from imageio import imread, imwrite
import cv2

import astra

import list_sinogram as LS
import FanPara_Transform as F2P
import Centering_Sino as CS
import ParaFan_Trans as P2F

from tqdm import tqdm, trange

import time

print("*** Feld-Kamp法再構成プログラム(by astra toolbox) ***")
input_dir = input("[入力]投影画像フォルダの名前を指定 >> ")
output_dir = input("[出力]断面画像フォルダの名前を指定 >> ")
os.makedirs(output_dir, exist_ok=True)
flg = int( input("グレースケール反転の有無を選択してください(1:反転する/0:反転しない) >> ") )

# Configuration.
distance_source_origin = 80  # 線源 --> 検査対象中心 距離 [mm]
distance_origin_detector = 40  # 検査対象中心 --> 検出器 [mm]
detector_pixel_size = 0.083 # 検出器の1pxあたりのサイズ [mm]
detector_rows = 1440  # 検出器の縦サイズ [pixels]
detector_cols = 1280  # 検出器の横サイズ [pixels]
num_of_projections = 360    # 投影数
angles = np.linspace(0, 2 * np.pi, num=num_of_projections, endpoint=False)

# Load projections.
# projections = np.zeros((detector_rows, num_of_projections, detector_cols))

files = glob.glob(input_dir + os.sep + "*.tif")
files.sort()

# サイノグラム作成に必要な変数の設定
in_sample = cv2.imread(files[0])
start = 0
fin = in_sample.shape[0] - 1

print(fin)

# 変数の更新
detector_cols = in_sample.shape[1]
detector_rows = in_sample.shape[0]
rate = int(input("基準画像サイズ(最も大きいもの、横ピクセル数) >> ")) / in_sample.shape[1]
detector_pixel_size *= rate

print({rate, detector_pixel_size, detector_rows, detector_cols})

# <再構成前処理 : 中心合わせ>
# 投影画像->サイノグラム
temp1 = LS.main(input_dir, start, fin, flg)

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
Centered_proj = np.zeros((len(temp2), temp2[0].shape[0], temp2[0].shape[1]))
for i in trange(len(temp2), desc='Sino -> Proj', leave=True):
	Centered_proj[i, :, :] = temp2[i]

print(Centered_proj.shape)

for i in trange(Centered_proj.shape[1], desc='CENT_Proj_Check', leave=True):
	cv2.imwrite("CENT_proj" + os.sep + "CentProj_" + str(i) + ".tif", Centered_proj[:,i,:].astype(np.uint16))

'''
for i in range(num_of_projections):
	im = cv2.imread(files[i],-1)
	if flg == 1:
		im = cv2.bitwise_not(im)
	im = im/65535.0
	projections[:, i, :] = im
'''
# 入力値に異常があるかどうか試してみた --> 関係なさそう
Centered_proj = Centered_proj / np.max(Centered_proj)

# Copy projection images into ASTRA Toolbox.
proj_geom = \
	astra.create_proj_geom('cone', 1, 1, int(detector_rows), int(detector_cols), angles,
	                       distance_source_origin /
	                       detector_pixel_size, distance_origin_detector / detector_pixel_size)
projections_id = astra.data3d.create('-sino', proj_geom, Centered_proj)

# Create reconstruction.
vol_geom = astra.creators.create_vol_geom(int(detector_cols), int(detector_cols),
                                          int(detector_rows))
reconstruction_id = astra.data3d.create('-vol', vol_geom, data=0)
alg_cfg = astra.astra_dict('FDK_CUDA')
alg_cfg['ProjectionDataId'] = projections_id
alg_cfg['ReconstructionDataId'] = reconstruction_id
algorithm_id = astra.algorithm.create(alg_cfg)

start = time.time()
astra.algorithm.run(algorithm_id)
process_time = time.time() - start
print("FDK全体の時間 : " + str(process_time * 1000)  + "[msec.]")

reconstruction = astra.data3d.get(reconstruction_id)

print(np.max(reconstruction))
print(np.min(reconstruction))

# Limit and scale reconstruction.
reconstruction[reconstruction < 0] = 0
reconstruction /= np.max(reconstruction)
reconstruction = reconstruction * 255
reconstruction = np.round(reconstruction).astype(np.uint8)

# Save reconstruction.
for i in trange(int(detector_rows), desc='Img_Output', leave=True):
	im = reconstruction[i, :, :]
	im = np.flipud(im)
	imwrite(join(output_dir, 'reco%04d.png' % i), im)

# Cleanup.
astra.algorithm.delete(algorithm_id)
astra.data3d.delete(reconstruction_id)
astra.data3d.delete(projections_id)
