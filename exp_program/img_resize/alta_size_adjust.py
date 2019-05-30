#-*- coding:utf-8 -*-
import cv2
import numpy as np
import os
import glob

def main(file_path,ratio_x,ratio_y):
	input_f = cv2.imread(file_path,-1)
	orgHeight, orgWidth = input_f.shape

	size = (int(orgWidth/ratio_x),int(orgHeight/ratio_y))
	print(size)

	#Resizing Image
	resized = cv2.resize(input_f, size)

	os.makedirs("resize", exist_ok = True)

	f_name, _ = os.path.splitext( os.path.basename(file_path) )

	cv2.imwrite("resize"+ os.sep +"resized_" + os.path.basename(f_name) + ".tif", resized.astype(np.uint16))

if __name__ == "__main__":
	#画像の読み込み(ディレクトリに画像は1枚だけ)
	folder = input("リサイズする画像のディレクトリを選択してください >>>")

	ratio_x = float(input("x方向の縮小倍率を指定してください >>> "))
	ratio_y = float(input("y方向の縮小倍率を指定してください >>> "))

	files = glob.glob(folder + os.sep + "*.tif")
	files.sort()

	print(files)

	#1枚ずつmain関数でリサイズ
	for f in files:
		print(os.path.basename(f))
		main(f,ratio_x,ratio_y)