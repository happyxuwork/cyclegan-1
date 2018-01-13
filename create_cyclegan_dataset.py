"""Create datasets for training and testing."""
import csv
import os
import random
import click
import cyclegan_datasets
"""
本程序的主要目的就是根据cyclegan_datasets中的配置
将需要输入的图片的位置信息，写入到一个.csv文件中
便于后面取图像数据
"""

def create_list(foldername, fulldir=True, suffix=".jpg"):
    """
    :param foldername: The full path of the folder.
    :param fulldir: Whether to return the full path or not.
    :param suffix: Filter by suffix.

    :return: The list of filenames in the folder with given suffix.
    """
    file_list_tmp = os.listdir(foldername)
    file_list = []
    if fulldir:
        for item in file_list_tmp:
            if item.endswith(suffix):
                file_list.append(os.path.join(foldername, item))
    else:
        for item in file_list_tmp:
            if item.endswith(suffix):
                file_list.append(item)
    return file_list


@click.command()
@click.option('--image_path_a',
              type=click.STRING,
              default='./input/horse2zebra/testA',
              help='The path to the images from domain_a.')
@click.option('--image_path_b',
              type=click.STRING,
              default='./input/horse2zebra/testB',
              help='The path to the images from domain_b.')
@click.option('--dataset_name',
              type=click.STRING,
              default='horse2zebra_test',
              help='The name of the dataset in cyclegan_dataset.')
@click.option('--do_shuffle',
              type=click.BOOL,
              default=False,
              help='Whether to shuffle images when creating the dataset.')
def create_dataset(image_path_a, image_path_b,
                   dataset_name, do_shuffle):
    list_a = create_list(image_path_a, True,
                         cyclegan_datasets.DATASET_TO_IMAGETYPE[dataset_name])
    list_b = create_list(image_path_b, True,
                         cyclegan_datasets.DATASET_TO_IMAGETYPE[dataset_name])

    output_path = cyclegan_datasets.PATH_TO_CSV[dataset_name]

    num_rows = cyclegan_datasets.DATASET_TO_SIZES[dataset_name]
    all_data_tuples = []
    for i in range(num_rows):
        all_data_tuples.append((
            list_a[i % len(list_a)],
            list_b[i % len(list_b)]
        ))
    if do_shuffle is True:
        random.shuffle(all_data_tuples)
    # if not os.path.exists(output_path):
    #     os.m
    #you konw if the open function do not have newline=''
    #the result csv may have the empty line
    with open(output_path, 'w',newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        #enumerate是一个枚举型，返回的是遍历索引，以及对应的遍历元素
        #enumerate(list,number),第二个参数数指定索引的起始值
        for data_tuple in enumerate(all_data_tuples):
            csv_writer.writerow(list(data_tuple[1]))


if __name__ == '__main__':
    create_dataset()
