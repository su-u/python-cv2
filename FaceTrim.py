#OpenCVのインポート
import cv2
import numpy as np
import glob
import os
 
#カスケード型分類器に使用する分類器のデータ（xmlファイル）を読み込み
HAAR_FILE="./haarcascade_frontalface_alt.xml"
cascade = cv2.CascadeClassifier(HAAR_FILE)

files = glob.glob("./img/SpringBase/*")
dirname = 'outDir'
if not os.path.exists(dirname):
    os.mkdir(dirname)


cut: int = 1
#画像ファイルの読み込み
for fname in files: 
    bgr = cv2.imread(fname, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    face_rects = cascade.detectMultiScale(gray, 1.1, 3)
    

#カスケード型分類器を使用して画像ファイルから顔部分を検出する
    face = cascade.detectMultiScale(gray)
 
 
#顔の座標を表示する
    print(face)
 
#顔部分を切り取る
    for x,y,w,h in face:
        face_cut = bgr[y:y+h, x:x+w]
 
#白枠で顔を囲む
    for x,y,w,h in face:
        cv2.rectangle(bgr,(x,y),(x+w,y+h),(255,255,255),2)
 

#画像の出力
    dir: str = f"./outDir/{str(cut)}.png"
    try:
        cv2.imwrite(dir, face_cut)
    except Exception as e:
        print(e)
    cut = cut + 1
