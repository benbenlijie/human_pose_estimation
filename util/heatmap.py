import numpy as np
import pickle
import os
from PIL import Image, ImageDraw
# from matplotlib import pyplot as plt
# import h5py

KEY_POINT_AMOUNT = 14
heat_map_storage = "../model_res/keypoint_heatmap.h5"
RAF_PAIR = [[0, 1], [1, 2], [3, 4], [4, 5], [6, 7], [7, 8], [9, 10], [10, 11], [12, 13], [13, 0], [13, 3], [13, 6], [13, 9]]


def gaussian(img, pt, sigma=4):
    # Draw a 2D gaussian
    pt = np.array(pt).round()
    # Check that any part of the gaussian is in-bounds
    ul = [int(pt[0] - 3 * sigma), int(pt[1] - 3 * sigma)]
    br = [int(pt[0] + 3 * sigma + 1), int(pt[1] + 3 * sigma + 1)]
    if (ul[0] > img.shape[1] or ul[1] >= img.shape[0] or
            br[0] < 0 or br[1] < 0):
        # If not, just return the image as is
        return img

    # Generate gaussian
    size = 6 * sigma + 1
    x = np.arange(0, size, 1, float)
    y = x[:, np.newaxis]
    x0 = y0 = size // 2
    # The gaussian is not normalized, we want the center value to equal 1
    g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))

    # Usable gaussian range
    g_x = max(0, -ul[0]), min(br[0], img.shape[1]) - ul[0]
    g_y = max(0, -ul[1]), min(br[1], img.shape[0]) - ul[1]
    # Image range
    img_x = max(0, ul[0]), min(br[0], img.shape[1])
    img_y = max(0, ul[1]), min(br[1], img.shape[0])

    img[img_y[0]:img_y[1], img_x[0]:img_x[1]] = g[g_y[0]:g_y[1], g_x[0]:g_x[1]]
    return img


def raf_vector(img, pt0, pt1, sigma=4):
    for pt in (pt0, pt1):
        ul = [int(pt[0] - 3 * sigma), int(pt[1] - 3 * sigma)]
        br = [int(pt[0] + 3 * sigma + 1), int(pt[1] + 3 * sigma + 1)]
        if (ul[0] > img.shape[1] or ul[1] >= img.shape[0] or
                    br[0] < 0 or br[1] < 0):
            # If not, just return the image as is
            return img
    pt0 = np.array(pt0)
    pt1 = np.array(pt1)
    vec = pt1 - pt0
    vec = vec / np.linalg.norm(vec)

    im = Image.new("F", (img.shape[1], img.shape[0]), color=0.)
    draw = ImageDraw.Draw(im)
    draw.line([*pt0, *pt1], fill=1., width=10)
    del draw
    result = np.array(im)
    img[:, :, 0] = vec[0] * result + img[:, :, 0]
    img[:, :, 1] = vec[1] * result + img[:, :, 1]
    return img


def extract_point_data(data_list):
    for i in range(0, len(data_list), 3):
        yield data_list[i:i+3]


def get_point_data_by_idx(data_list, idx, stride=3):
    if 0 <= idx < (len(data_list) / stride):
        return data_list[idx * stride: (idx+1) * stride]


def generate_raf(img_path, annotations, new_size=None):
    img = Image.open(img_path)
    ori_size = img.size
    if new_size is None:
        new_size = img.size
        xrate, yrate = 1., 1.
    else:
        xrate, yrate = new_size[1] * 1. / ori_size[1], new_size[0] * 1. / ori_size[0]
    # print(annotations)
    raf = np.zeros([new_size[1], new_size[0], (len(RAF_PAIR) + 1) * 2])
    person_amount = len(annotations['keypoint_annotations'].keys())
    cach_raf = np.zeros([new_size[1], new_size[0], len(RAF_PAIR) * 2, person_amount])
    person_idx = -1
    for (key, points) in annotations['keypoint_annotations'].items():
        person_idx += 1
        idx = 0
        for [a, b] in RAF_PAIR:
            data_a = get_point_data_by_idx(points, a)
            data_b = get_point_data_by_idx(points, b)
            if not (data_a[2] == 3 or data_b[2] == 3):
                cach_raf[:, :, idx:idx+2, person_idx] = raf_vector(cach_raf[:, :, idx:idx+2, person_idx],
                                                                   np.multiply(data_a[:2], [yrate, xrate]),
                                                                   np.multiply(data_b[:2], [yrate, xrate]))
            idx += 2
    raf[:, :, :-2] = np.mean(cach_raf, axis=-1)
    raf[:, :, -2] = np.mean(raf[:, :, np.arange(0, len(RAF_PAIR) * 2, 2)], axis=-1)
    raf[:, :, -1] = np.mean(raf[:, :, np.arange(1, len(RAF_PAIR) * 2, 2)], axis=-1)
#    show(img, raf)
    return img, raf


def generate_heatmap(img_path, annotations, new_size=None):
    img = Image.open(img_path)
    ori_size = img.size
    if new_size is None:
        new_size = img.size
        xrate, yrate = 1., 1.
    else:
        xrate, yrate = new_size[1] * 1. / ori_size[1], new_size[0] * 1. / ori_size[0]
    # print(annotations)
    heatmap = np.zeros([new_size[1], new_size[0], KEY_POINT_AMOUNT * 2 + 3])
    person_amount = len(annotations['keypoint_annotations'].keys())
    cach_heatmap = np.zeros([new_size[1], new_size[0], KEY_POINT_AMOUNT * 2, person_amount])
    person_idx = -1
    for (key, points) in annotations['keypoint_annotations'].items():
        person_idx += 1
        idx = -1
        for point_data in extract_point_data(points):
            if point_data[2] == 3:
                pass
            else:
                cach_heatmap[:, :, idx + point_data[2], person_idx] = \
                    gaussian(cach_heatmap[:, :, idx + point_data[2], person_idx],
                             [point_data[0] * yrate, point_data[1] * xrate])
            idx += 2
    heatmap[:, :, :KEY_POINT_AMOUNT * 2] = np.max(cach_heatmap,  axis=-1)
    heatmap[:, :, -1] = np.max(heatmap[:, :, np.arange(0, KEY_POINT_AMOUNT * 2, 1)], axis=-1)
    heatmap[:, :, -2] = np.max(heatmap[:, :, np.arange(1, KEY_POINT_AMOUNT * 2, 2)], axis=-1)
    heatmap[:, :, -3] = np.max(heatmap[:, :, np.arange(0, KEY_POINT_AMOUNT * 2, 2)], axis=-1)
#    show(img, raf)
    return img, heatmap

"""
def show(img, heatmap):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(22, 10))
    ax1.imshow(img)
    ax1.imshow(heatmap[:, :, -3], alpha=0.5)
    ax2.imshow(img)
    ax2.imshow(heatmap[:, :, -2], alpha=0.5)
    ax3.imshow(img)
    ax3.imshow(heatmap[:, :, -1], alpha=0.5)
    plt.show()
"""

if __name__ == '__main__':
    with open("../model_res/keypoint_annotation.pkl", mode="rb") as f:
        anno_dict = pickle.load(f)
    new_size = [512, 512]
    for img in os.listdir('../sample_img'):
        img_id = os.path.splitext(img)[0]

        annos = anno_dict[img_id]
        img_path = "../sample_img/{}".format(img)
        img, heatmap = generate_heatmap(img_path, annos, new_size=new_size)
        # show(img.resize(new_size, Image.ANTIALIAS), heatmap)

        img, raf = generate_raf(img_path, annos, new_size=new_size)
        # show(img.resize(new_size, Image.ANTIALIAS), raf)
        break
