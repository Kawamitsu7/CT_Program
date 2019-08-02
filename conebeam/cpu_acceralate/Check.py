import cv2
import numpy as np

#以下各種チェック用関数、デバッグに使う

#イメージ中の最大階調値を出力
def check_maxpixel(I_im):
	print(np.amax(I_im))
	quit()

#イメージサイズを表示
def check_dim_of_mat(I_im):
	print(I_im.shape)

#イメージを表示してプログラムを終了する : 引数に0を入れるとプログラム継続
def check_and_quit(I_im,flg):
	cv2.namedWindow("Plz press any key.", cv2.WINDOW_NORMAL)
	cv2.imshow("Plz press any key.",I_im)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	if flg == 1:
		quit()