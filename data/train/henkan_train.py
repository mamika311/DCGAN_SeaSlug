import os,glob
import numpy as np
from tqdm import tqdm
from keras.preprocessing.image import load_img,img_to_array
from keras.utils import np_utils
from sklearn import model_selection
from PIL import Image

#クラスを配列に格納
classes = ["SeaSlug_train1", "SeaSlug_train2"]

num_classes = len(classes)
img_size = 128
color=False

#画像の読み込み
#最終的に画像、ラベルはリストに格納される

temp_img_array_list=[]
temp_index_array_list=[]
for index,classlabel in enumerate(classes):
    photos_dir = "./" + classlabel
    #globでそれぞれのクラスの画像一覧を取得
    img_list = glob.glob(photos_dir + "/*.jpg")
    for img in tqdm(img_list):
        temp_img=load_img(img,grayscale=color,target_size=(img_size, img_size))
        temp_img_array=img_to_array(temp_img)
        temp_img_array_list.append(temp_img_array)
        temp_index_array_list.append(index)
        # 回転の処理
        for angle in range(-20,20,5):
            # 回転
            img_r = temp_img.rotate(angle)
            data = np.asarray(img_r)
            temp_img_array_list.append(data)
            temp_index_array_list.append(index)
            # 反転
            img_trans = img_r.transpose(Image.FLIP_LEFT_RIGHT)
            data = np.asarray(img_trans)
            temp_img_array_list.append(data)
            temp_index_array_list.append(index)

            X=np.array(temp_img_array_list)
            Y=np.array(temp_index_array_list)

np.save("./SeaSlug_train_img_128RGB.npy", X)
np.save("./SeaSlug_train_index_128RGB.npy", Y)
