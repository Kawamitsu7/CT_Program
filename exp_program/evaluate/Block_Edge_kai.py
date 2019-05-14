import cv2
import numpy as np
import glob
import os
import matplotlib.pyplot as plt
import math

import pandas as pd

drawing = False # true if mouse is pressed
mode = True # if True, draw rectangle. Press 'm' to toggle to curve
ix,iy,ox,oy = -1,-1,-1,-1
iteration_flg = 0
tilt_th_trace = []
left, right = -500, -500
num_data = -1

def interpolation(I_im, theta_f, Xf):
	x,y = fold_back(theta_f,Xf,I_im.shape)

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

	left_up = dst[y,x] * (1-diff_x) * (1-diff_y)
	left_down = dst[y+1,x] * (1-diff_x) * diff_y
	right_up = dst[y,x+1] * diff_x * (1-diff_y)
	right_down = dst[y+1,x+1] * diff_x * diff_y

	return left_up + left_down + right_up + right_down

def fold_back(y,x,img_size):
	if x >= img_size[1]:
		#print("over flow x : "+str(x))
		x = img_size[1] - 1
		
		if x >= img_size[1]:
			print("x error :" + str(x))
	elif x < 0:
		#print("under flow x : "+str(x))
		x = 0

	if y >= img_size[0]:
		#print("over flow y : "+str(y))
		y = img_size[0] - 1
	elif y < 0:
		#print("under flow y : "+str(y))
		y = 0

	return x,y

def main(f):
	px_size = 83.0 / 1000
	bin_size = px_size
	global iteration_flg
	global ix,iy,ox,oy
	global tilt_th_trace

	# mouse callback function
	def draw_circle(event,x,y,flags,param):
		global ix,iy,ox,oy,drawing,mode

		if event == cv2.EVENT_LBUTTONDOWN:
			drawing = True
			ix,iy = x,y

		elif event == cv2.EVENT_LBUTTONUP:
			drawing = False
			ox,oy = x,y
			if mode == True:
				cv2.rectangle(img,(ix,iy),(x,y),(0,255,0),1)
			else:
				cv2.circle(img,(x,y),5,(0,0,255),-1)

	print("処理中 : " + os.path.basename(f))

	img = cv2.imread(f,-1)
	copy = cv2.imread(f,-1)

	px_size = px_size / (img.shape[0]/1280)

	if iteration_flg == 0:
		ROI_flg = input("ROIの設定方法を選択 : 0 -> D&D / 1 -> 数値入力 \n >> ")

		if ROI_flg == "0":
			# ~ROI指定部分~
			#改良予定 : 元画像から何回もROIを設定できるようにする(今は黒線のガイドが残ってしまう)
			cv2.namedWindow('image',cv2.WINDOW_NORMAL)
			cv2.setMouseCallback('image',draw_circle)
			while(1):
				cv2.imshow('image',img)
				k = cv2.waitKey(1) & 0xFF
				if k == ord('m'):
					mode = not mode
				elif k == 27:	#escでROI指定終了
					break
			cv2.destroyAllWindows()

		else:
			print("比較対象と画像サイズが変わらないことを確認して下さい")
			#ROI手入力
			ix = int(input("左上のx座標 : ix >> "))
			iy = int(input("左上のy座標 : iy >> "))
			ox = int(input("右下のx座標 : ox >> "))
			oy = int(input("右下のy座標 : oy >> "))

	# ~ROI部分トリミング~

	crop = copy[iy:oy + 1,ix:ox +1]
	print("ix : {}pixel".format(ix))
	print("iy : {}pixel".format(iy))
	print("ox : {}pixel".format(ox))
	print("oy : {}pixel".format(oy))

	# ~ピクセルプロファイルデータプロット~
	prof = np.zeros((crop.shape[0],crop.shape[1],3))

	for i in range (crop.shape[0]):
		for j in range (crop.shape[1]):
			prof[i,j,0] = j
			prof[i,j,1] = crop[i,j]
			prof[i,j,2] = i


	plt.figure()
	plt.scatter(prof[:,:,0],prof[:,:,1],s=0.4, c=prof[:,:,2])
	plt.colorbar()
	plt.xlabel('pixels')# 横軸のラベル名
	plt.ylabel('pixel intensity')	# 縦軸のラベル名
	plt.grid(True)

	plt.show()

	# ~エッジの座標をもとにエッジそろえ~
	# 一番下の行に合わせる
	tilt_th = 2.0
	edge_roop_flg = True
	trace_pointer = 0

	fixed_prof = prof

	while(edge_roop_flg):
		for h in range (crop.shape[0]):
			fixed_prof[h,:,0] = fixed_prof[h,:,0] + (crop.shape[0] - h) * math.tan(math.radians(tilt_th))
		
		if iteration_flg == 0:
			plt.figure()
			plt.scatter(fixed_prof[:,:,0],fixed_prof[:,:,1],s=0.4, c=fixed_prof[:,:,2])
			plt.colorbar()
			plt.xlabel('pixels')# 横軸のラベル名
			plt.ylabel('pixel intensity')	# 縦軸のラベル名
			plt.grid(True)

			plt.show()

		if iteration_flg == 0:
			edge_roop_flg = ( input("エッジ調整を継続しますか : y/n? >>") == "y" )

			if edge_roop_flg == 1:
				tilt_th = float(input("エッジの傾斜角度を入力(float) >>"))
				tilt_th_trace.append(tilt_th)
		else:
			if trace_pointer == len(tilt_th_trace) - 1:
				edge_roop_flg = False
			else:
				tilt_th = tilt_th_trace[trace_pointer]
				trace_pointer += 1

	#thetaの関係ないデータ配列へと置き換える
	data = np.zeros((2,crop.shape[0] * crop.shape[1]))
	for th in range (crop.shape[0]):
		data[0, crop.shape[1]*th : crop.shape[1]*(th+1)] = fixed_prof[th,:,0] * px_size
		data[1, crop.shape[1]*th : crop.shape[1]*(th+1)] = fixed_prof[th,:,1]
	data = data[:,np.argsort(data[0,:])]
	#print(data[0,:])
	if iteration_flg == 0:
		plt.figure()
		plt.scatter(data[0,:],data[1,:],s=0.4)
		plt.xlabel('distance of center(mm)')# 横軸のラベル名
		plt.ylabel('pixel intensity')	# 縦軸のラベル名
		plt.grid(True)

		plt.show()

	# ~bin処理~
	binned_data = []
	binned_r = []
	start = math.floor(min(data[0,:])/bin_size)*bin_size
	pt = start
	temp = 0
	counter = 0
	#print(start)

	for i in range (data.shape[1]):
		if data[0,i] < pt + bin_size:
			temp += data[1,i]
			counter += 1
		else:
			binned_r.append(pt)
			binned_data.append(temp/counter)
			pt = pt + bin_size
			temp = data[0,i]
			counter = 1

	if iteration_flg == 0:
		plt.figure()
		plt.scatter(binned_r,binned_data,s=0.4)
		plt.xlabel('distance of center(mm)')# 横軸のラベル名
		plt.ylabel('pixel intensity')	# 縦軸のラベル名
		plt.grid(True)

		plt.show()

	# ~微分 : LSF取得~
	diff_r = binned_r[0:len(binned_r)-1]
	diff_data = np.diff(binned_data,1)

	plt.figure()
	plt.scatter(diff_r,diff_data,s=0.4,label='diffed')
	plt.xlabel('distance of center(mm)')# 横軸のラベル名
	plt.ylabel('pixel intensity')	# 縦軸のラベル名
	plt.grid(True)

	plt.show()

	# ~zeroing~
	global left, right
	if iteration_flg == 0:
		left = float(input("左側でゼロ処理する境界の座標を指定 >>>"))
		right = float(input("右側でゼロ処理する境界の座標を指定 >>>"))

	search_flg = 0
	for i in range(len(diff_r)):
		if search_flg == 0 and diff_r[i] > left:
			search_flg = 1
			left_zeroing = i-1
		elif search_flg == 1 and diff_r[i] > right:
			search_flg = 2
			right_zeroing = i

	diff_data[0:left_zeroing + 1] = 0
	diff_data[right_zeroing : len(diff_data)] = 0

	if iteration_flg == 0:
		plt.figure()
		plt.scatter(diff_r,diff_data,s=0.4,label='zeroing')
		plt.xlabel('distance of center(mm)')# 横軸のラベル名
		plt.ylabel('pixel intensity')	# 縦軸のラベル名
		plt.grid(True)

		plt.show()

	# ~データ数調整~
	# データの個数を2のべき乗に調整

	print("現在のデータ数は{}".format(len(diff_data)))
	global num_data
	if iteration_flg ==0:
		num_data = int(input("データ数を選択(2のべき乗) >>>"))
	zero_left = 1.0 * ( num_data - len(diff_data) ) / 2
	zero_left = int(zero_left)
	print(num_data)
	for i in range(zero_left):
		diff_data = np.insert(diff_data, 0, 0)
		diff_data = np.append(diff_data,0)
	if len(diff_data) % 2 == 1:
		diff_data = np.append(diff_data,0)
	print("現在のデータ数は{}".format(len(diff_data)))

	# ~MTF算出~
	fft_data = np.fft.fft(diff_data,len(diff_data))
	mtf_blurred_data = np.abs(fft_data)/np.abs(fft_data[0]) 
	freq = np.linspace(0, 1.0/bin_size, len(fft_data))
	mtf_data = []

	#np.seterr(all = "print")

	for i in range (len(freq)):
		#c_bin = (math.pi * freq[f] * bin_size)/math.sin(math.pi * freq[f] * bin_size)
		c_bin = 1/np.sinc(freq[i] * bin_size)
		c_diff = c_bin
		mtf_data.append( mtf_blurred_data[i] * c_bin * c_diff )

	plt.figure()
	plt.scatter(freq[:int(len(freq)/2)],mtf_data[:int(len(mtf_data)/2)],label='MTF_blurred')
	plt.xlabel('spatial frequancy(cycles/mm)')# 横軸のラベル名
	plt.ylabel('MTF')	# 縦軸のラベル名
	plt.grid(True)

	plt.show()

	#pandasによるデータのcsv出力部分
	output_flg = int(input("csvに出力しますか : Yes -> 1 / No -> 0 \n >>"))
	#データの用意
	if output_flg == 1:
		hol_data = pd.DataFrame(freq[:int(len(freq)/2)],columns = ['spatial frequancy(cycles/mm)'])
		ver_data = pd.DataFrame(mtf_data[:int(len(mtf_data)/2)],columns = ['MTF(Score)'])
		data = pd.concat([hol_data,ver_data],axis=1)
		#データをcsvに書き出し
		f_name, _ = os.path.splitext( os.path.basename(f) )
		data.to_csv("MTF_Score_" + os.path.basename(f_name) + ".csv")

	#img = cv2.circle(crop,center,radius,(65535,65535,65535),2)
	os.makedirs("Crop_img", exist_ok = True)
	cv2.imwrite("Crop_img"+ os.sep +"Croped.tif", crop.astype(np.uint16))
	#cv2.imwrite("Crop_img"+ os.sep +"check_th.tif", analyze_prof.astype(np.uint16))

	if iteration_flg == 0:
		iteration_flg = 1

if __name__ == "__main__":
	# ~ファイル読み込み~
	folder = input("解析対象画像のディレクトリを選択してください >>>")
	files = glob.glob(folder + os.sep + "*.tif")
	files.sort()
	#print(files)
	for f in files:
		main(f)