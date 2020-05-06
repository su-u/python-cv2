#OpenCVのインポート
import cv2
import numpy as np
import glob
import os
 
#カスケード型分類器に使用する分類器のデータ（xmlファイル）を読み込み
HAAR_FILE="./haarcascade_frontalface_alt.xml"
cascade = cv2.CascadeClassifier(HAAR_FILE)

# files = glob.glob("./img/same/*")
files = glob.glob("./img/SpringBase/*")
dirname = 'outDir'
if not os.path.exists(dirname):
    os.mkdir(dirname)


#画像ファイルの読み込み
for fname in files:
  bgr = cv2.imread(fname, cv2.IMREAD_COLOR)
  gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
  
  face = cascade.detectMultiScale(gray)

  if len(face) <= 0:
    print("顔認証できず(´･ω･｀)")
    print(fname)
    continue

  count: int = 0
  for x, y, w, h in face:
    face_cut = bgr[y: y + h, x: x + w]
    cv2.rectangle(bgr, (x, y), (x + w, y + h), (255, 255, 255), 2)
    try:
        dir: str = f"{dirname}/{os.path.basename(fname)}-{count}.jpg"
        cv2.imwrite(dir, face_cut)
        count += 1
    except Exception as e:
        print(e)
