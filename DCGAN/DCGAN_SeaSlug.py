# # 使用可能メモリの確認
# !free -h

# # メモリの増量
# [_ for _ in range(10000000000)]

# # マウント
# from google.colab import drive
# drive.mount('/content/drive')


# 必要ライブラリのインストール
# %matplotlib inline
import pickle as pkl
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# 画像の読み込み
train_img = np.load("SeaSlug/data/train/SeaSlug_train_img_128RGB.npy")
train_index = np.load("SeaSlug/data/train/SeaSlug_train_index_128RGB.npy")
test_img = np.load("SeaSlug/data/test/SeaSlug_test_img_128RGB.npy")
test_index = np.load("SeaSlug/data/test/SeaSlug_test_index_128RGB.npy")
valid_img = np.load("SeaSlug/data/valid/SeaSlug_valid_img_128RGB.npy")
valid_index = np.load("SeaSlug/data/valid/SeaSlug_valid_index_128RGB.npy")

# int型に変換
train_img1 = train_img.astype(np.int64)
test_img1 = test_img.astype(np.int64)
valid_img1 = valid_img.astype(np.int64)

# 4次元の配列を3次元に
i=0
train_img2 = train_img1[i,:,:,]
test_img2 = test_img1[i,:,:,]
valid_img2 = valid_img1[i,:,:,]

# データをランダムに取り出して表示
idx = np.random.randint(train_img1.shape[0], size=25)
fig, axes = plt.subplots(5, 5,sharex=True, sharey=True, figsize=(9,9))
for ii, ax in zip(idx, axes.flatten()):
    ax.imshow(train_img1[ii], aspect="equal")
plt.subplots_adjust(wspace=0, hspace=0)
imgplot = plt.imshow(train_img2)

# 本物データをGeneratorで生成したデータのスケールを-1~1で揃える
def scale(x, feature_ranges=(-1, 1)):
    # 0~1に変換
    x = ((x - x.min()) / (255 - x.min()))

    # -1~1に変換
    min, max = feature_ranges
    x = x * (max - min) + min
    return x

# データセットのクラスを定義
class Dataset:
    # val_fracでテストデータを学習中と学習後用に分離する
    # スケール関数は上のものを使うためscale_func=None
    def __init__(self, train_img2, test_img2, valid_img2, train_index, valid_index, test_index, shuffle= False, scale_func=None):
        self.test_x, self.valid_x = test_img2, valid_img2
        self.test_y, self.valid_y = test_index, valid_index
        self.train_x, self.train_y = train_img2, train_index

        if scale_func is None:
            self.scaler = scale
        else:
            self.scaler = scale_func
        self.shuffle = shuffle

    # ミニバッチ生成の定義
    def batches(self, batch_size):
        if self.shuffle:
            idx = np.arange(len(dataset.train_x))
            np.random.shuffle(idx)
            self.train_x = self.train_x[idx]
            self.train_y = self.train_y[idx]

        n_batches = len(self.train_y) // batch_size
        for ii in range(0, len(self.train_y), batch_size):
            x = self.train_x[ii:ii+batch_size]
            y = self.train_y[ii:ii+batch_size]

            yield self.scaler(x), y

# プレースホルダーを生成
def model_inputs(real_dim, z_dim):
    inputs_real = tf.placeholder(tf.float32, (None, *real_dim), name="input_real")
    inputs_z = tf.placeholder(tf.float32, (None, z_dim), name="input_z")

    return inputs_real, inputs_z

# ジェネレーターの生成(128x128)
def generator(z, output_dim, reuse=False, alpha=0.2, training=True):
    with tf.variable_scope("generator", reuse=reuse):
        # 1層目
        x1 = tf.layers.dense(z, 4*4*512)
        # 2次元データを1次元に変換する
        x1 = tf.reshape(x1, (-1, 4, 4, 512))
        # データの傾きを少なくなるように調整する
        x1 = tf.layers.batch_normalization(x1, training=training)
        # 活性関数の適用
        x1 = tf.maximum(alpha * x1, x1)
        # この時点でサイズは4x4x512

        # 2層目
        x2 = tf.layers.conv2d_transpose(x1, 256, 5, strides=2, padding="same")
        x2 = tf.layers.batch_normalization(x2, training=training)
        x2 = tf.maximum(alpha * x2, x2)
        # この時点でサイズは8x8x256

        # 3層目
        x3 = tf.layers.conv2d_transpose(x2, 128, 5, strides=2, padding="same")
        x3 = tf.layers.batch_normalization(x3, training=training)
        x3 = tf.maximum(alpha * x3, x3)
        # この時点でサイズは16x16x128

        # 4層目
        x4 = tf.layers.conv2d_transpose(x3, 64, 5, strides=2, padding="same")
        x4 = tf.layers.batch_normalization(x4, training=training)
        x4 = tf.maximum(alpha * x4, x4)
        # この時点でサイズは32x32x64

        # 5層目
        x5 = tf.layers.conv2d_transpose(x4, 32, 5, strides=2, padding="same")
        x5 = tf.layers.batch_normalization(x5, training=training)
        x5 = tf.maximum(alpha * x5, x5)
        # この時点でサイズは64x64x32

        logits = tf.layers.conv2d_transpose(x5, output_dim, 5, strides=2, padding="same")
        # この時点でサイズは128x128x3

        # 正解データと揃えるために-1~1で変換
        out = tf.tanh(logits)

        return out

# ディスクリミネーターの生成(128x128)
def discriminator(x, reuse=False, alpha=0.2):
    with tf.variable_scope("discriminator", reuse=reuse):
        x1 = tf.layers.conv2d(x, 32, 5, strides=2, padding="same")
        x1 = tf.maximum(alpha * x1, x1)
        x1_drop = tf.nn.dropout(x1, 0.5)
        # この時点でサイズは64x64x32

        x2 = tf.layers.conv2d(x1_drop, 64, 5, strides=2, padding="same")
        x2 = tf.layers.batch_normalization(x2, training=True)
        x2 = tf.maximum(alpha * x2, x2)
        x2_drop = tf.nn.dropout(x2, 0.5)
        # この時点でサイズは32x32x64

        x3 = tf.layers.conv2d(x2_drop, 128, 5, strides=2, padding="same")
        x3 = tf.layers.batch_normalization(x3, training=True)
        x3 = tf.maximum(alpha * x3, x3)
        x3_drop = tf.nn.dropout(x3, 0.5)
        # この時点でサイズは16x16x128

        x4 = tf.layers.conv2d(x3_drop, 256, 5, strides=2, padding="same")
        x4 = tf.layers.batch_normalization(x4, training=True)
        x4 = tf.maximum(alpha * x4, x4)
        x4_drop = tf.nn.dropout(x4, 0.5)
        # この時点でサイズは8x8x256

        x5 = tf.layers.conv2d(x4_drop, 512, 5, strides=2, padding="same")
        x5 = tf.layers.batch_normalization(x5, training=True)
        x5 = tf.maximum(alpha * x5, x5)
        x5_drop = tf.nn.dropout(x5, 0.5)
        # この時点でサイズは4x4x512

        # 全結合層で1列に並べる
        flat = tf.reshape(x5_drop, (-1, 4*4*512))
        logits = tf.layers.dense(flat, 1)
        logits_drop = tf.nn.dropout(logits, 0.5)
        # sigmoidを適用して0~1で出力する
        out = tf.sigmoid(logits_drop)

        return out, logits

# 損失関数の生成
def model_loss(input_real, input_z, output_dim, alpha=0.2):
    g_model = generator(input_z, output_dim, alpha=alpha)
    d_model_real , d_logits_real = discriminator(input_real, alpha=alpha)
    d_model_fake , d_logits_fake = discriminator(g_model, reuse=True, alpha=alpha)

    # 本物データを判別するときの損失(本物を正しく判別できるか)
    d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_real, labels=tf.ones_like(d_model_real)))
    # 偽物データを判別するときの損失(偽物を正しく判別できるか)
    d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_fake, labels=tf.zeros_like(d_model_fake)))
    # 偽物データを入れた時に本物と判別するかの設定
    g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_fake, labels=tf.ones_like(d_model_fake)))

    # 上の2つを合算して最終的なd_lossを出力
    d_loss = d_loss_real + d_loss_fake

    return d_loss, g_loss

# 最適化の設定
def model_opt(d_loss, g_loss, learning_rate, beta1):
    # パラメータを格納する変数を生成
    t_vars = tf.trainable_variables()
    d_vars = [var for var in t_vars if var.name.startswith("discriminator")]
    g_vars = [var for var in t_vars if var.name.startswith("generator")]

    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        d_train_opt = tf.train.AdamOptimizer(learning_rate, beta1=beta1).minimize(d_loss, var_list=d_vars)
        g_train_opt = tf.train.AdamOptimizer(learning_rate, beta1=beta1).minimize(g_loss, var_list=g_vars)

    return d_train_opt, g_train_opt

# モデルの定義
class GAN:
    def __init__(self, real_size, z_size, learning_rate, alpha=0.2, beta1=0.5, verbose=0):
        tf.reset_default_graph()
        # 上のmodel_inputsから持ってくるだけ
        self.input_real, self.input_z = model_inputs(real_size, z_size)
        # 上のmodel_lossから持ってくるだけ
        self.d_loss, self.g_loss = model_loss(self.input_real, self.input_z, real_size[2], alpha=alpha)
        # 上のmodel_optから持ってくるだけ
        self.d_opt, self.g_opt = model_opt(self.d_loss, self.g_loss, learning_rate, beta1)

# 生成した画像を表示する関数の定義
def view_samples(epoch, samples, nrows, ncols, figsize=(5,5)):
    fig, axes = plt.subplots(figsize=figsize, nrows=nrows, ncols=ncols,sharey=True, sharex=True)
    for ax, img in zip(axes.flatten(), samples[epoch]):
        ax.axis('off')
        img = ((img - img.min())*255 / (img.max() - img.min())).astype(np.int64)
        im = ax.imshow(img, aspect='equal')

    plt.subplots_adjust(wspace=0, hspace=0)

    return fig, axes

# トレーニングの関数を定義
def train(net, dataset, epochs, batch_size, print_every=10, show_every=100, figsize=(5,5)):
    # 途中のパラメータの保存
    saver = tf.train.Saver()
    # サンプル生成
    sample_z = np.random.uniform(-1, 1, size=(72, z_size))

    samples, losses = [], []
    steps = 0

    with tf.Session() as sess:
        # 変数のリセット
        sess.run(tf.global_variables_initializer())
        # saver.restore(sess, "SeaSlug/save_file/256_0.001/SeaSlug0_50(256, 0.0001).ckpt")
        for e in range(epochs):
            # バッチで取り出してパラメータの更新を行う
            for x, y in dataset.batches(batch_size):
                # for文のたびにstep数を1増加
                steps += 1

                batch_z = np.random.uniform(-1, 1, size=(batch_size, z_size))

                # _では処理を実行しても値を返さない
                # net.d_optの計算にfeed_dictを用いる
                _ = sess.run(net.d_opt, feed_dict={net.input_real: x, net.input_z: batch_z})
                _ = sess.run(net.g_opt, feed_dict={net.input_z: batch_z, net.input_real: x})

                # 途中経過を表示する処理
                if steps % print_every == 0:
                    train_loss_d = net.d_loss.eval({net.input_z: batch_z, net.input_real: x})
                    train_loss_g = net.g_loss.eval({net.input_z: batch_z})

                    print("Epoch {}/{}: ".format(e+1, epochs),
                         "D Loss: {:.4f}  ".format(train_loss_d),
                         "G Loss: {:.4f}  ".format(train_loss_g))

                    # train_loss_dとtrian_loss_gをlossesの一番後ろに格納する
                    losses.append((train_loss_d, train_loss_g))

                if steps % show_every == 0:
                    gen_samples = sess.run(generator(net.input_z, 3, reuse=True, training=False),
                                          feed_dict={net.input_z: sample_z})
                    samples.append(gen_samples)
                    _ = view_samples(-1, samples, 5, 5, figsize=figsize)
                    plt.show()


        # saver.save(sess, "SeaSlug/save_file/50_100/SeaSlug50_100(256, 0.0001).ckpt")

    return losses, samples

# パラメータの初期化
real_size = (128, 128, 3)
z_size = 100
learning_rate = 0.0001
batch_size = 256
epochs = 150
alpha = 0.2
beta1 = 0.5

net = GAN(real_size, z_size, learning_rate, alpha=alpha, beta1=beta1)

# # Hyperdashのインストール
# !pip install hyperdash
# from hyperdash import monitor_cell
# !hyperdash login --email

# Hyperdashの使用
from tensorflow.keras.callbacks import Callback
from hyperdash import Experiment

class Hyperdash(Callback):
    def __init__(self, entries, exp):
        super(Hyperdash, self).__init__()
        self.entries = entries
        self.exp = exp

    def on_epoch_end(self, epoch, logs=None):
        for entry in self.entries:
            log = logs.get(entry)
            if log is not None:
                self.exp.metric(entry, log)

exp = Experiment("SeaSlug50_100(256, 0.0001)epoc")
hd_callback = Hyperdash(["val_loss", "loss", "val_accuracy", "accuracy"], exp)

# トレーニングの実行
dataset = Dataset(train_img1, test_img1, valid_img1, train_index, valid_index, test_index)
losses, samples = train(net, dataset, epochs, batch_size, figsize=(9, 9))

exp.end()

# ロスのプロット
fig, ax = plt.subplots()
losses = np.array(losses)
plt.plot(losses.T[0], label='D', alpha=0.5)
plt.plot(losses.T[1], label='G', alpha=0.5)
plt.title('Training Loss')
plt.legend()

# # 学習ファイルの保存
# from google.colab import files
# files.download( "/content/SeaSlug500_550(256, 0.0001).ckpt.meta" )
