# -*- coding: UTF-8 -*-
'''
@author: xuqiang
'''
# import numpy as np
# a = np.zeros((50,1,2,2,3))
# print(a)

import tensorflow as tf
import numpy as np
import tensorlayer

'''
进行批加载
'''
# def generate_data():
#     num = 25
#     label = np.asarray(range(0, num))
#     images = np.random.random([num, 5, 5, 3])
#     print('label size :{}, image size {}'.format(label.shape, images.shape))
#     return label, images
#
# def get_batch_data():
#     label, images = generate_data()
#     images = tf.cast(images, tf.float32)
#     label = tf.cast(label, tf.int32)
#     input_queue = tf.train.slice_input_producer([images, label], shuffle=False)
#     image_batch, label_batch = tf.train.batch(input_queue, batch_size=10, num_threads=1, capacity=64)
#     return image_batch, label_batch
#
# image_batch, label_batch = get_batch_data()
# with tf.Session() as sess:
#     coord = tf.train.Coordinator()
#     threads = tf.train.start_queue_runners(sess, coord)
#     i = 0
#     try:
#         while not coord.should_stop():
#             image_batch_v, label_batch_v = sess.run([image_batch, label_batch])
#             i += 1
#             for j in range(10):
#                 print(image_batch_v.shape, label_batch_v[j])
#     except tf.errors.OutOfRangeError:
#         print("done")
#     finally:
#         coord.request_stop()
#     coord.join(threads)




'''
进行文件的读写操作
'''
import csv
# def write_to_csv(output_path_file_name):
#     a = []
#     dict = {}
#     for i in range(100):
#         a.append(str(i)+"yes")
#         dict[str(i)] = str(i)+"yes"
#     with open(output_path_file_name,'w',newline='') as file:
#         csv_writer = csv.writer(file)
#         csv_writer.writerow(dict)
#         # for text in enumerate(dict):
#         #     csv_writer.writerow(text)
# if __name__ == "__main__":
#     write_to_csv('yes.csv')


"""
load_sample by the queue
"""
import tensorflow as tf
import cyclegan_datasets
import model
import matplotlib.pyplot as plt


"""
load data的操作
"""
def _load_samples(csv_name, image_type):
    filename_queue = tf.train.string_input_producer(
        [csv_name])

    reader = tf.TextLineReader()
    _, csv_filename = reader.read(filename_queue)

    record_defaults = [tf.constant([], dtype=tf.string),
                       tf.constant([], dtype=tf.string)]

    filename_i, filename_j = tf.decode_csv(
        csv_filename, record_defaults=record_defaults)

    file_contents_i = tf.read_file(filename_i)
    file_contents_j = tf.read_file(filename_j)
    if image_type == '.jpg':
        image_decoded_A = tf.image.decode_jpeg(
            file_contents_i, channels=model.IMG_CHANNELS)
        image_decoded_B = tf.image.decode_jpeg(
            file_contents_j, channels=model.IMG_CHANNELS)
    elif image_type == '.png':
        image_decoded_A = tf.image.decode_png(
            file_contents_i, channels=model.IMG_CHANNELS, dtype=tf.uint8)
        image_decoded_B = tf.image.decode_png(
            file_contents_j, channels=model.IMG_CHANNELS, dtype=tf.uint8)

    return image_decoded_A, image_decoded_B


def load_data(dataset_name, image_size_before_crop,
              do_shuffle=True, do_flipping=False):
    """

    :param dataset_name: The name of the dataset.
    :param image_size_before_crop: Resize to this size before random cropping.
    :param do_shuffle: Shuffle switch.
    :param do_flipping: Flip switch.
    :return:
    """
    print("============yes===========")
    if dataset_name not in cyclegan_datasets.DATASET_TO_SIZES:
        raise ValueError('split name %s was not recognized.'
                         % dataset_name)

    csv_name = cyclegan_datasets.PATH_TO_CSV[dataset_name]
    #进行数据的加载
    image_i, image_j = _load_samples(
        csv_name, cyclegan_datasets.DATASET_TO_IMAGETYPE[dataset_name])
    inputs = {
        'image_i': image_i,
        'image_j': image_j
    }

    # Preprocessing:
    #进行图片大小的调整，调整为image_size_before_crop=286大小
    inputs['image_i'] = tf.image.resize_images(
        inputs['image_i'], [image_size_before_crop, image_size_before_crop])
    inputs['image_j'] = tf.image.resize_images(
        inputs['image_j'], [image_size_before_crop, image_size_before_crop])

    if do_flipping is True:
        #进行图片的翻转--左右
        inputs['image_i'] = tf.image.random_flip_left_right(inputs['image_i'])
        inputs['image_j'] = tf.image.random_flip_left_right(inputs['image_j'])
    #按特定的大小对调整后的图片进行修剪
    inputs['image_i'] = tf.random_crop(
        inputs['image_i'], [model.IMG_HEIGHT, model.IMG_WIDTH, 3])
    inputs['image_j'] = tf.random_crop(
        inputs['image_j'], [model.IMG_HEIGHT, model.IMG_WIDTH, 3])

    inputs['image_i'] = tf.subtract(tf.div(inputs['image_i'], 127.5), 1)
    inputs['image_j'] = tf.subtract(tf.div(inputs['image_j'], 127.5), 1)

    inputs['images_i'], inputs['images_j'] = tf.train.batch([inputs['image_i'], inputs['image_j']], 1)

    #Batch
    if do_shuffle is True:
        inputs['images_i'], inputs['images_j'] = tf.train.shuffle_batch(
            [inputs['image_i'], inputs['image_j']], 1, 5000, 100)
    else:
        inputs['images_i'], inputs['images_j'] = tf.train.batch(
            [inputs['image_i'], inputs['image_j']], 1)
    #注意:这里的inputs可是有四张图片啊images_i，images_j，image_i,image_j
    #print(inputs['images_j'].eval())
    return inputs

import layers
import matplotlib.pyplot as plt
def discriminator_tf(inputdisc, name="discriminator"):
    with tf.variable_scope(name):
        f = 4
        o_c1 = layers.general_conv2d(inputdisc, 64, f, f, 2, 2,0.02, "SAME", "c1", do_norm=False,relufactor=0.2)
        o_c2 = layers.general_conv2d(o_c1, 64 * 2, f, f, 2, 2,0.02, "SAME", "c2", relufactor=0.2)
        o_c3 = layers.general_conv2d(o_c2, 64 * 4, f, f, 2, 2,0.02, "SAME", "c3", relufactor=0.2)
        o_c4 = layers.general_conv2d(o_c3, 64 * 8, f, f, 1, 1, 0.02, "SAME", "c4", relufactor=0.2)
        o_c5 = layers.general_conv2d(o_c4, 1, f, f, 1, 1, 0.02,"SAME", "c5", do_norm=False, do_relu=False)
        return o_c5

if __name__ == "__main__":
    init = (tf.global_variables_initializer(),
            tf.local_variables_initializer())
    inputs = load_data("horse2zebra_train", 286, do_shuffle=True, do_flipping=False)
    prob = discriminator_tf(inputs['images_j'])
    with tf.Session() as sess:
        sess.run(init)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        inputs = sess.run(inputs)
        # print("==========image_i============")
        # print(inputs['image_i'].shape)
        # print(inputs['image_i'])
        # plt.imshow(inputs['image_i'])
        # plt.show()
        # print("==========image_i============")
        # print("==========image_i还原============")
        # img = sess.run(tf.cast(((inputs['image_i'] + 1) * 127.5), dtype=tf.uint8))
        # print(img)
        # plt.imshow(img)
        # plt.show()
        # print("==========image_i还原============")
        #
        # print("==========images_i============")
        #
        # inputs['images_i'] = np.reshape(inputs['images_i'], (256, 256, 3))
        # print(inputs['images_i'].shape)
        # print(inputs['images_i'])
        # plt.imshow(inputs['images_i'])
        # plt.show()
        # print("==========images_i============")
        # print("==========images_i还原============")
        # img = sess.run(tf.cast(((inputs['images_i'] + 1) * 127.5), dtype=tf.uint8))
        # print(img)
        # plt.imshow(img)
        # plt.show()

        print("==========images_i还原============")

        # prob = sess.run([prob],feed_dict={inputs['images_j']:inputs['images_j']})
        #feed_dict = {
            #                 self.input_a:
            #                     inputs['images_i'],
        print(discriminator_tf(inputs['images_j']))








