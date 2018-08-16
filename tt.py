 # -*- coding: utf-8 -*-

__author__ = 'Pan'


# train module train all the trainable ops in graph

import tensorflow as tf
import data_download
from gennet.encoder_decoder import *
from facenet import face_net
from emonet import emotion_net
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import cv2
import datetime
import time
import sys
import os


def image_variable(filename):
    f_data = cv2.imread(filename)
    data = []
    for i in range(0, 160):
        data.append([])
        for j in range(0, 160):
            data[-1].append([f_data[i][j][2], f_data[i][j][1], f_data[i][j][0]])
            # data[-1].append([f_data[i][j][0]])
    a = [data]
    return tf.Variable(a, dtype=tf.float32, name='GImage')


def get_noise_image(width, height, channel):
    return tf.get_variable('GImage', [1, width, height, channel], tf.float32, tf.truncated_normal_initializer(mean=128, stddev=50))

def get_gram(feature_map, size):
    vector = tf.reshape(feature_map, [size * size, -1])
    vetor_t = tf.transpose(vector)
    return tf.matmul(vetor_t, vector, name='Gram')

def get_gram_loss(sess, tensor_name, size, name):
    with tf.variable_scope(name):
        feature_map = sess.graph.get_tensor_by_name(tensor_name)
        feature_map1, feature_map2, feature_map3 = tf.split(feature_map, [1, 1, 1], axis=0)
        gram1 = get_gram(feature_map1, size)
        gram2 = get_gram(feature_map3, size)
        temp_loss = tf.reduce_mean(tf.square(tf.subtract(gram1, gram2)))
        return temp_loss


def train(epoch, batches, weight1, weight2, weight3, lr, layer_name, dir_name):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        img1_raw_data = tf.gfile.FastGFile('data/wyz160.jpg', 'rb').read()
        img2_raw_data = tf.gfile.FastGFile('data/xtf160.jpg', 'rb').read()
        image_batch1 = tf.reshape(tf.cast(tf.image.decode_jpeg(img1_raw_data), tf.float32), [1, 160, 160, 3], name='input1')
        image_batch2 = tf.reshape(tf.cast(tf.image.decode_jpeg(img2_raw_data), tf.float32), [1, 160, 160, 3], name='input2')
        # image_batch2 = tf.reshape(tf.cast(tf.image.decode_jpeg(img2_raw_data), tf.float32), [1, 160, 160, 3], name='input2')
        
        gen_batch1 = image_variable('data/xtf160.jpg')
        # gen_batch1 = get_noise_image(160, 160, 3)
        with tf.variable_scope("normalize1"):
            split_list = []
            for i in range(0, batches):
                split_list.append(1)
            imgs1 = tf.split(gen_batch1, split_list, axis=0)
            imgs2 = tf.split(image_batch1, split_list, axis=0)
            imgs3 = tf.split(image_batch2, split_list, axis=0)
            norm_imgs1 = []
            norm_imgs2 = []
            norm_imgs3 = []
            for i in range(0, batches):
                norm_imgs1.append(tf.reshape(tf.image.per_image_standardization(tf.reshape(imgs1[i], [160, 160, 3])), [1, 160, 160, 3]))
                norm_imgs2.append(tf.reshape(tf.image.per_image_standardization(tf.reshape(imgs2[i], [160, 160 ,3])), [1, 160, 160, 3]))
                norm_imgs3.append(tf.reshape(tf.image.per_image_standardization(tf.reshape(imgs3[i], [160, 160 ,3])), [1, 160, 160, 3]))
            norm1 = tf.concat(norm_imgs1, axis=0)
            norm2 = tf.concat(norm_imgs2, axis=0)
            norm3 = tf.concat(norm_imgs3, axis=0)

        group_batch1 = tf.concat([norm1, norm2, norm3], axis=0)

        # # 人脸识别网络
        face_net.load_model(group_batch1)
        # face_output = sess.graph.get_tensor_by_name('import/embeddings:0')
        # face_output = sess.graph.get_tensor_by_name('import/InceptionResnetV1/Repeat_2/block8_5/Relu:0')
        # face_output = sess.graph.get_tensor_by_name('import/InceptionResnetV1/Conv2d_2b_3x3/Relu:0')
        face_output = sess.graph.get_tensor_by_name(layer_name + ':0')
        face_output_1, face_output_2, face_output_3 = tf.split(face_output, [batches, batches, batches])

        # group for emotion recognition
        # group_batch2 = tf.concat([gen_batch1, image_batch2], axis=0)
        # 表情识别网络
        # 将图片转为1维的灰度图
        # reduced_group_batch = tf.reshape(tf.div(tf.reduce_sum(group_batch2, axis=3), 3.0), shape=[-1, 160, 160, 1])
        # emotion_net.load_model(reduced_group_batch)
        # emotion_net.load_model(group_batch2)
        # emotion_output = sess.graph.get_tensor_by_name('import_1/emonet/block4/Block4/conv4/Conv4/Conv:0')
        # emotion_output = sess.graph.get_tensor_by_name('import/emonet/conv1/Block4/Conv4:0')
        # emotion_output_1, emotion_output_2 = tf.split(emotion_output, [batches, batches])

        # norm_emotion_output = tf.nn.l2_normalize(emotion_output, 1, 1e-10, name='emotion_embeddings')
        # emotion_output_1, emotion_output_2 = tf.split(norm_emotion_output, [batches, batches])
        # cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=emotion_output_1, labels=emotion_output_2, name='cross_entropy')
        #
        # Face original loss
        # with tf.variable_scope('Oringial_loss'):
        #     ct = tf.constant(128.0)
        #     norm1_1d = tf.reshape(tf.divide(gen_batch1, ct), [160*160*3])
        #     norm2_1d = tf.reshape(tf.divide(image_batch1, ct), [160*160*3])
        #     loss0 = tf.reduce_mean(tf.square(tf.subtract(norm1_1d, norm2_1d)))
        #     loss0_weight = tf.constant(weight1)
        #     loss0 = tf.multiply(loss0, loss0_weight, name='orignal_loss')
        # with tf.variable_scope('Oringial_loss2'):
        #     ct = tf.constant(128.0)
        #     norm1_1d = tf.reshape(tf.divide(gen_batch1, ct), [160*160*3])
        #     norm2_1d = tf.reshape(tf.divide(image_batch2, ct), [160*160*3])
        #     loss1 = tf.reduce_mean(tf.square(tf.subtract(norm1_1d, norm2_1d)))
        #     loss1_weight = tf.constant(weight2)
        #     loss1 = tf.multiply(loss1, loss1_weight, name='orignal_loss')
        # 人脸识别损失
        with tf.variable_scope('Face_loss'):
            temp_loss = tf.reduce_mean(tf.square(tf.subtract(face_output_1, face_output_2)))
            loss1_weight = tf.constant(weight1)
            loss1 = tf.multiply(temp_loss, loss1_weight, name='face_loss')
        # 纹理损失
        with tf.variable_scope('Texture_loss'):
            gram_loss1 = get_gram_loss(sess, 'import/InceptionResnetV1/Conv2d_1a_3x3/Relu:0', 79, 'gram_1a')
            gram_loss2 = get_gram_loss(sess, 'import/InceptionResnetV1/Conv2d_2b_3x3/Relu:0', 77, 'gram_2b')
            gram_loss3 = get_gram_loss(sess, 'import/InceptionResnetV1/Conv2d_3b_1x1/Relu:0', 38, 'gram_3b')
            gram_loss4 = get_gram_loss(sess, 'import/InceptionResnetV1/Conv2d_4b_3x3/Relu:0', 17, 'gram_4b')
            gram_loss5 = get_gram_loss(sess, 'import/InceptionResnetV1/Repeat/block35_5/Relu:0', 17, 'gram_block35')
            gram_loss = gram_loss1 + gram_loss2 + gram_loss3 + gram_loss4 + gram_loss5
            gram_weight = tf.constant(weight2)
            gram_loss = tf.multiply(gram_loss, gram_weight, name='gram_loss')
        # 浅层信息损失
        # lower_output = sess.graph.get_tensor_by_name('import/InceptionResnetV1/Conv2d_2b_3x3/Relu:0')
        # lower_output_1, lower_output_2, lower_output_3 = tf.split(lower_output, [batches, batches, batches])
        with tf.variable_scope('content_loss'):
            # temp_loss = tf.reduce_mean(tf.square(tf.subtract(norm1, norm2)))
            temp_loss = tf.reduce_mean(tf.abs(tf.subtract(norm1, norm3)))
            loss3_weight = tf.constant(weight3)
            loss3 = tf.multiply(temp_loss, loss3_weight, name='face_loss')
        # 表情识别损失
        # with tf.variable_scope('Emotion_loss'):
        #     temp_loss = tf.reduce_mean(tf.square(tf.subtract(emotion_output_1, emotion_output_2)))
        #     loss4_weight = tf.constant(1.0)
        #     loss4 = tf.multiply(temp_loss, loss4_weight, name='emotion_loss')
        
        # tf.summary.scalar('loss0', loss0)
        tf.summary.scalar('identification_loss2', loss1)
        tf.summary.scalar('gram_loss2', gram_loss)
        tf.summary.scalar('content_loss2', loss3)
        # # 组合3个损失
        total_loss = tf.add(tf.add(loss1, gram_loss), loss3, name='total_loss')
        # total_loss = tf.add(total_loss, loss4, name='total_loss2')
        tf.summary.scalar('total_loss', total_loss)
        
        # 做梯度优化
        train_step = tf.train.AdamOptimizer(lr).minimize(total_loss)
        merge = tf.summary.merge_all()

        writer = tf.summary.FileWriter('./train_summary', sess.graph)
        phase_train = sess.graph.get_tensor_by_name('import/phase_train:0')
        # keep_prob = sess.graph.get_tensor_by_name('import_1/emonet/keep_prob:0')

        tf.local_variables_initializer().run()
        tf.global_variables_initializer().run()

        start_time = time.time()

        # a = sess.run(emotion_output, feed_dict={keep_prob:1.0})
        # print a
        for i in range(0, 10000):
            _, total_summary, _loss, img_data = sess.run([train_step, merge, total_loss, gen_batch1], feed_dict={phase_train:False})
            print('Current {0} loss: {1} gen_image: {2}'.format(i, _loss, img_data[0][0][0][0]))
            writer.add_summary(total_summary, i)
            if (i+1) % 100 == 0:
                end_time = time.time()
                timedelta = end_time - start_time
                print(timedelta)
                image = Image.new('RGB', (160, 160), (255, 255, 255))
                draw = ImageDraw.Draw(image)
                # for x in range(160):
                #     for y in range(160):
                #         draw.point((y, x), fill=(img_data[0][x][y][2], img_data[0][x][y][1], img_data[0][x][y][0]))
                #         # draw.point((y, x), fill=(img_data[0][x][y][0], img_data[0][x][y][0], img_data[0][x][y][0]))
                # image.save('result/edit/' + dir_name + '/gen_image_' + str(timedelta)[:5] + '.bmp', 'bmp')
                for x in range(160):
                    for y in range(160):
                        draw.point((y, x), fill=(img_data[0][x][y][0], img_data[0][x][y][1], img_data[0][x][y][2]))
                        # draw.point((y, x), fill=(img_data[0][x][y][0], img_data[0][x][y][0], img_data[0][x][y][0]))
                image.save('result/edit2/' + dir_name + '/gen_image_' + str(timedelta)[:5] + '.bmp', 'bmp')

        writer.close()


if __name__ == '__main__':
    # learning_rate = float(sys.argv[1])
    # layer = sys.argv[2]
    # name = sys.argv[3]
    # if not os.path.exists('result/edit/' + name):
    #     os.makedirs('result/edit/' + name)
    # train(10, 1, 100.0, 0.0000001, 1.0, learning_rate, layer, name)
    # 
    train(10, 1, 100.0, 0.0000001, 1.5, 0.5, 'import/InceptionResnetV1/Repeat_2/block8_5/Relu', 'wyz_xtf3')
    