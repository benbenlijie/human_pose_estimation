import threading

from util.heatmap import *
import tensorflow as tf
import json
import numpy as np


class DataIterator(object):
    def __init__(self, annotation_file, img_file_format, batch_size, data_shape, shuffle=False, seed=None, part=-1):

        with tf.gfile.GFile(annotation_file, "r") as f:
            self.annotations = json.load(f)
        if part != -1:
            choose = np.random.permutation(len(self.annotations))[:part]
            self.annotations = list(np.array(self.annotations)[choose])
        self.img_file_format = img_file_format
        self.batch_size = batch_size
        self.batch_index = 0
        self.data_shape = data_shape

        self.N = len(self.annotations)

        self.total_batches = self.N // self.batch_size
        self.lock = threading.Lock()
        self.shuffle = shuffle
        self.seed = seed

    def reset(self):
        self.batch_index = 0

    def next(self):
        with self.lock:
            if self.batch_index == 0:
                if self.seed is not None:
                    np.random.seed(self.seed + self.total_batches)
                self.index_array = (np.random.permutation(self.N) if self.shuffle
                                    else np.arange(self.N))

            batches_x, batches_y1, batches_y2 = \
                [], [], []

            start = self.batch_index * self.batch_size
            end = (self.batch_index + 1) * self.batch_size

            idxes = self.index_array[start:end]


            # add to batch all samples from batch_keys

            for idx in idxes:
                annotation = self.annotations[idx]
                # image
                img_file_path = self.img_file_format.format(annotation["image_id"])
                if os.path.exists(img_file_path) is False:
                    continue
                img, heatmap = generate_heatmap(img_file_path, annotation, new_size=[128, 128])
                img, raf = generate_raf(img_file_path, annotation, new_size=[128, 128])
                data_img = np.array(img.resize(self.data_shape, Image.ANTIALIAS))
                batches_x.append(data_img[np.newaxis, ...])

                batches_y1.append(raf[np.newaxis, ...])
                batches_y2.append(heatmap[np.newaxis, ...])

            self.batch_index += 1

            if self.batch_index == self.total_batches:
                self.batch_index = 0
            if batches_x is []:
                return [], []
            batch_x = np.concatenate(batches_x)
            batch_y1 = np.concatenate(batches_y1)
            batch_y2 = np.concatenate(batches_y2)

            return [batch_x], \
                   [batch_y1, batch_y2,
                    batch_y1, batch_y2,
                    batch_y1, batch_y2,
                    batch_y1, batch_y2,
                    batch_y1, batch_y2,
                    batch_y1, batch_y2]

    def __iter__(self):
        return self

    def __next__(self, *args, **kwargs):
        return self.next(*args, **kwargs)


if __name__ == '__main__':
    data_dir = "../../train_data/ai_challenger_keypoint_train_20170909/"
    train_di = DataIterator(data_dir + "/keypoint_train_annotations_20170909.json",
                            data_dir + "/keypoint_train_images_20170902/{}.jpg",
                            batch_size=256,
                            data_shape=[512, 512])
    for i in range(train_di.total_batches+1):
        train_di.next()
        print("operate {}: {} images".format(i+1, (i+1) * train_di.batch_size))
