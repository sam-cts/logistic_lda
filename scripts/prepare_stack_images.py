"""
Downloads Pinterest data and writes it into tf-records format
"""

import json
import os
import urllib.request
from collections import defaultdict
import glob
import cv2

import bson
import numpy as np
import tensorflow as tf
from PIL import Image
from sklearn.model_selection import StratifiedShuffleSplit

CAT_PATH = 'data/chiptype/categories.txt'

IMG_DATA_PATH = '/mnt/mlserver/datasets/chips/20200804_individual_stack/vgg_data'
TF_DATA_PATH_TRAIN = 'data/tf_chiptype_train'
TF_DATA_PATH_TEST = 'data/tf_chiptype_test'

TARGET_IMG_SIZE = (224, 224)

if not os.path.isdir(TF_DATA_PATH_TRAIN):
    os.makedirs(TF_DATA_PATH_TRAIN)

if not os.path.isdir(TF_DATA_PATH_TEST):
    os.makedirs(TF_DATA_PATH_TEST)

if not os.path.isdir(IMG_DATA_PATH):
    os.makedirs(IMG_DATA_PATH)


def pad(img):
    new_im = Image.new("RGB", TARGET_IMG_SIZE)
    new_im.paste(img, ((TARGET_IMG_SIZE[0] - img.size[0]) // 2,
                       (TARGET_IMG_SIZE[1] - img.size[1]) // 2))
    return new_im


def make_tf_record(imagePaths, target, maxTypes, idxs, topics):
    print('writing %s tf records' % target)
    data = []
    for idx in idxs:
        # for pin_id in board2pin_ids[board_id]:
        # img_path = '%s/%s.jpg' % (IMG_DATA_PATH, pin_id)
        imagePath = imagePaths[idx]
        maxType = topics[maxTypes[idx]]
        if os.path.isfile(imagePath):
            try:
                pin_id = 0
                img = np.array(Image.open(imagePath), dtype='uint8')
                
                img.shape 
                for i in range(16):
                    for j in range(16):
                        r1 = i*14
                        r2 = i*14+14
                        c1 = j*14
                        c2 = j*14+14
                        subsection = img[r1:r2, c1:c2, :]

                        subsection = cv2.resize(subsection, (224, 224), cv2.INTER_LINEAR)
                        data.append((imagePath, str(pin_id), subsection, maxType))
                        pin_id += 1

            except:
                print('skipped image', imagePath)

    tf_records_path = TF_DATA_PATH_TRAIN if target == 'train' else TF_DATA_PATH_TEST
    write_tfrecords(tf_records_path, data)
    write_meta_info(tf_records_path, data=data, target=target, topics=topics)


def write_tfrecords(target_path, data, max_items_per_tfrecord=20000):
    count = 0
    shard = 0

    writer = tf.python_io.TFRecordWriter(
        os.path.join(target_path, 'part-{0:05}.tfrecord').format(shard))

    # write TFRecord files
    for author_id, item_id, img, label in data:
        # create writers of multiple part files
        if count % max_items_per_tfrecord == 0:
            if shard > 0:
                writer.close()
            writer = tf.python_io.TFRecordWriter(
                os.path.join(target_path, 'part-{0:05}.tfrecord').format(shard))
            shard += 1

        img_h, img_w, img_c = img.shape[0], img.shape[1], img.shape[2]
        # create data point
        example = tf.train.Example(
            features=tf.train.Features(
                feature={
                    'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img.tostring()])),
                    'image_height': tf.train.Feature(int64_list=tf.train.Int64List(value=[img_h])),
                    'image_width': tf.train.Feature(int64_list=tf.train.Int64List(value=[img_w])),
                    'image_depth': tf.train.Feature(int64_list=tf.train.Int64List(value=[img_c])),
                    'author_id': tf.train.Feature(bytes_list=tf.train.BytesList(value=[author_id.encode()])),
                    'author_topic': tf.train.Feature(bytes_list=tf.train.BytesList(value=[label.encode()])),
                    'item_id': tf.train.Feature(bytes_list=tf.train.BytesList(value=[item_id.encode()])),
                    'item_topic': tf.train.Feature(bytes_list=tf.train.BytesList(value=[b''])),
                }))

        # write to training or test set
        writer.write(example.SerializeToString())
        count += 1

    writer.close()


def write_meta_info(target_path, data, target, topics):
    # write meta information into JSON files
    author_ids = list(set([entry[0] for entry in data]))

    with open(target_path.rstrip('/') + '.json', 'w') as handle:
        json.dump({
            'features': {
                'image': {'shape': [], 'dtype': 'string'},
                'image_height': {'shape': [], 'dtype': 'int64'},
                'image_width': {'shape': [], 'dtype': 'int64'},
                'image_depth': {'shape': [], 'dtype': 'int64'},
                'author_id': {'shape': [], 'dtype': 'string'},
                'author_topic': {'shape': [], 'dtype': 'string'},
                'item_id': {'shape': [], 'dtype': 'string'},
                'item_topic': {'shape': [], 'dtype': 'string'},
            },
            'meta': {
                'topics': topics,
                'author_ids': author_ids,
            }
        }, handle)

def load_chip_dataset_vgg():
    # assert os.path.exists(IMG_DATA_PATH)
    path = os.path.join(IMG_DATA_PATH, "*.json")
    print(path)
    annoPaths = glob.glob(path)
    print(len(annoPaths))
    imagePaths = []
    maxTypes = []
    # this is the board
    for i, annoPath in enumerate(annoPaths):
        if i > 1000:
            break
        with open(annoPath, 'r') as jFile:
            annoFile = json.load(jFile)

        anno = list(annoFile.values())[0]
        imagePaths.append(os.path.join(IMG_DATA_PATH, anno['filename']))

        coloursDict = anno['regions'][0]['region_attributes']['colour_count']

        colours = list(coloursDict.values())
        # pop unsure 
        colours.pop(-2)
        maxTypes.append(colours.index(max(colours)))

    return imagePaths, maxTypes



def make_new_dataset(test_size=0.5):
    with open(CAT_PATH) as f:
        content = f.readlines()
        content = [x.strip() for x in content]

    # stackImgs is  a list of single stack image
    # maxTypes is a list of chip type that the stack contain the most of
    imagePaths, maxTypes = load_chip_dataset_vgg()
    print(f"number of images: {len(imagePaths)}")
    print(f"number of types: {len(maxTypes)}")

    # split into train and test sets
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=42)
    for t, v in sss.split(imagePaths, maxTypes):
        train_idxs, test_idxs = list(t), list(v)

    # train set
    print('n train images', len(train_idxs))
    make_tf_record(imagePaths, 'train', maxTypes, train_idxs, content)

    # test set
    print('n test images', len(test_idxs))
    make_tf_record(imagePaths, 'test', maxTypes, test_idxs, content)


if __name__ == '__main__':
    make_new_dataset(test_size=0.2)
