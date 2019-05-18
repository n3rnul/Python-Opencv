# # エッジを検出
# import cv2
# import numpy as np
#
# img = cv2.imread('./doc/akiyama.png', 0)
#
# edges = cv2.Canny(img, 100, 200)
#
# cv2.imshow('edges', edges)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# 顔を認識して画像を貼り付けるだけ
import cv2
from PIL import Image

# カスケードファイルを指定して、分類機を作成
cascade_file = "./doc/haarcascade_frontalface_alt.xml"
cascade = cv2.CascadeClassifier(cascade_file)

# 画像を読み込み、グレイスケールに変換
img = cv2.imread("./doc/akiyama.png")
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 顔検出
face_list = cascade.detectMultiScale(img_gray, minSize=(150, 150))
if len(face_list) == 0:
    quit()
print("face_list:", face_list)  # 2次元リストになっていることに注意

# わかりやすいように、face_listの値をまとめておく
x = face_list[0][0]  # X座標
y = face_list[0][1]  # Y座標
w = face_list[0][2]  # 横幅
h = face_list[0][3]  # 縦幅

# PILで画像を開く
im1 = Image.open('./doc/akiyama.png')
im2 = Image.open('./doc/p20a.png')

# 顔に合ったサイズに添付する画像をリサイズする
im22 = im2.resize((w, h))
im22.save('./doc/p20a.png')

# 原本はそのままにしておくため、コピーを加工する
back_im = im1.copy()

# 背景透過のためのsplit()
back_im.paste(im22, (x, y), im22.split()[3])

# 保存する
back_im.save('./doc/new_akiyama.png')
