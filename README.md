# DCGAN_SeaSlug
[深層学習で綺麗なウミウシを生成する](https://qiita.com/mamika311/items/75bc33cdeea17612bcd0)
こちらに詳細を記載しております。

# 開発環境
- Python
- Tensorflow
- Google Colaboratory

# 制作物概要
TensorFlowを用いて畳み込みニューラルネットワークを活用したDCGANです。  
大量の画像を学習し，新たな画像を生成します。  
今回はウミウシの画像を生成したためSea_Slugという名前にしております。  
![image.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/402915/6689bb9d-beba-5d2c-207c-1e05a1d1da70.png)![image.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/402915/772dfffd-9f25-0cb6-cc8b-6c8afcf79d80.png)![image.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/402915/a59c94a4-0fb2-9b07-43ef-daeb9b4dcfd9.png)

# フォルダ構成
- DCGAN
  - メインのプログラム
- data
  - 用意した`.npy`ファイルを格納するフォルダ
- download_img
  - 画像をダウンロードするプログラム
  - 今回はFlickrのAPIとGoogleImageCrawlerを用いて画像を収集
- save_file
  - メインプログラムの学習結果が保存されるフォルダ
