# リファレンスサイト "https://tomroelandts.com/articles/astra-toolbox-tutorial-reconstruction-from-projection-images-part-2"

import numpy as np
import os
import glob
from os.path import join
from imageio import imread, imwrite

import astra

print("*** Feld-Kamp法再構成プログラム(by astra toolbox) ***")
input_dir = input("[入力]投影画像フォルダの名前を指定 >> ")
output_dir = input("[出力]断面画像フォルダの名前を指定 >> ")
os.makedirs(output_dir, exist_ok=True)
flg = int( input("グレースケール反転の有無を選択してください(1:反転する/0:反転しない) >> ") )

# Configuration.
distance_source_origin = 80  # 線源 --> 検査対象中心 距離 [mm]
distance_origin_detector = 40  # 検査対象中心 --> 検出器 [mm]
detector_pixel_size = 0.083 * 8  # 検出器の1pxあたりのサイズ [mm]
detector_rows = 180  # 検出器の縦サイズ [pixels]
detector_cols = 160  # 検出器の横サイズ [pixels]
num_of_projections = 360    # 投影数
angles = np.linspace(0, 2 * np.pi, num=num_of_projections, endpoint=False)

# Load projections.
projections = np.zeros((detector_rows, num_of_projections, detector_cols))

files = glob.glob(input_dir + os.sep + "*.tif")
files.sort()

for i in range(num_of_projections):
	im = imread(files[i]).astype(float)
	if flg == 1:
		im = 65535 - im
	im /= 65535
	projections[:, i, :] = im

# Copy projection images into ASTRA Toolbox.
proj_geom = \
	astra.create_proj_geom('cone', 1, 1, detector_rows, detector_cols, angles,
	                       distance_source_origin /
	                       detector_pixel_size, distance_origin_detector / detector_pixel_size)
projections_id = astra.data3d.create('-sino', proj_geom, projections)

# Create reconstruction.
vol_geom = astra.creators.create_vol_geom(detector_cols, detector_cols,
                                          detector_rows)
reconstruction_id = astra.data3d.create('-vol', vol_geom, data=0)
alg_cfg = astra.astra_dict('FDK_CUDA')
alg_cfg['ProjectionDataId'] = projections_id
alg_cfg['ReconstructionDataId'] = reconstruction_id
algorithm_id = astra.algorithm.create(alg_cfg)
astra.algorithm.run(algorithm_id)
reconstruction = astra.data3d.get(reconstruction_id)

# Limit and scale reconstruction.
reconstruction[reconstruction < 0] = 0
reconstruction /= np.max(reconstruction)
reconstruction = np.round(reconstruction * 255).astype(np.uint8)

# Save reconstruction.
for i in range(detector_rows):
	im = reconstruction[i, :, :]
	im = np.flipud(im)
	imwrite(join(output_dir, 'reco%04d.png' % i), im)

# Cleanup.
astra.algorithm.delete(algorithm_id)
astra.data3d.delete(reconstruction_id)
astra.data3d.delete(projections_id)
