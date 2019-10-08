# リファレンスサイト "https://tomroelandts.com/articles/astra-toolbox-tutorial-reconstruction-from-projection-images-part-1"

import numpy as np
import os
from os.path import join
from imageio import get_writer

import astra


# ***プログラムインターフェース***
print("*** 3D投影データの作成プログラム ***")
output_dir = input("画像出力先フォルダの名前を指定 >> ")
os.makedirs(output_dir, exist_ok=True)


# ***仮想スキャナーのパラメータ設定***
distance_source_origin = 80  # 線源 --> 検査対象中心 距離 [mm]
distance_origin_detector = 40  # 検査対象中心 --> 検出器 [mm]
detector_pixel_size = 0.083 * 8  # 検出器の1pxあたりのサイズ [mm]
detector_rows = 180  # 検出器の縦サイズ [pixels]
detector_cols = 160  # 検出器の横サイズ [pixels]
num_of_projections = 360    # 投影数
angles = np.linspace(0, 2 * np.pi, num=num_of_projections, endpoint=False)


# ***ファントム(検査対象)作成部分***
# astraにおけるボリュームデータのジオメトリ情報作成
vol_geom = astra.creators.create_vol_geom(detector_cols, detector_cols, detector_rows)
# np配列でファントムの形状を設定
# <<<<任意の投影画像を作る場合いじるのはここ>>>>
phantom = np.zeros((detector_rows, detector_cols, detector_cols))
hb = 26  # Height of beam [pixels].
wb = 38  # Width of beam [pixels].
offset_h = 0
offset_w = 2
phantom[detector_rows // 2 - hb // 2 + offset_h: detector_rows // 2 + hb // 2 + offset_h,
detector_cols // 2 - wb // 2 + offset_w: detector_cols // 2 + wb // 2 + offset_w,
detector_cols // 2 - wb // 2 + offset_w: detector_cols // 2 + wb // 2 + offset_w] = 1
# ファントムのボリュームデータをastraの形式で作成
phantom_id = astra.data3d.create('-vol', vol_geom, data=phantom)


# ***投影画像作成部***
# 注意 : 角度が大きくなる --> オブジェクトが時計回りに回転 (線源は反時計回りと等価?)
# astraにおける投影データのジオメトリ情報作成
proj_geom = \
	astra.create_proj_geom('cone', 1, 1, detector_rows, detector_cols, angles,
	                       distance_source_origin /
	                       detector_pixel_size, distance_origin_detector / detector_pixel_size)
# ボリュームデータから投影画像の計算
projections_id, projections = \
	astra.creators.create_sino3d_gpu(phantom_id, proj_geom, vol_geom)
# 画像の正規化
projections /= np.max(projections)


# ***ポアソンノイズの適用***
projections = np.random.poisson(projections * 10000) / 10000
projections[projections > 1.1] = 1.1
projections /= 1.1


# ***投影データ保存***
projections = np.round(projections * 65535).astype(np.uint16)
for i in range(num_of_projections):
	projection = projections[:, i, :]
	with get_writer(join(output_dir, 'proj%04d.tif' % i)) as writer:
		writer.append_data(projection, {'compress': 9})


# astra用データのクリーンアップ
astra.data3d.delete(projections_id)
astra.data3d.delete(phantom_id)