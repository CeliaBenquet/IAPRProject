import os

import cv2
import numpy as np
import skimage.morphology
import matplotlib.pyplot as plt
from scipy.spatial import distance

def normalize(frame):
    fact = 255/frame[70:100, 40:100].mean()
    normalized_im = np.clip(fact*frame, 0, 255).astype('uint8')
    return normalized_im

def threshold(frame):
    return frame.mean(axis=2) < 175

def thresholdArrow(frame):
    r = frame[:,:,2]
    g = frame[:,:,1]
    b = frame[:,:,0]

    return np.logical_and(np.logical_and(r > 100, b < 60), g < 60)


def bounding_box(frame):
    rows = np.any(frame, axis=1)
    cols = np.any(frame, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return cmin, rmin, cmax, rmax


def merge_two_bboxes(A, B):
    x11, y11, x21, y21 = A
    x12, y12, x22, y22 = B
    return (min(x11, x12), min(y11, y12), max(x21, x22), max(y21, y22))


def merge_bboxes(bboxes, to_merge):
    for merge in to_merge:
        A = bboxes[merge[0]]
        B = bboxes[merge[1]]
        bboxes[merge[0]] = merge_two_bboxes(A, B)
        bboxes[merge[1]] = merge_two_bboxes(A, B)
    return bboxes


def bounding_boxes(frame):
    labels, nb_labels = skimage.morphology.label(frame, return_num=True)

    frame_size = frame.shape[0] * frame.shape[1]

    bounding_boxes = []

    for i in range(nb_labels):
        area = np.sum(labels == i) / frame_size
        if area > 1e-3 or area < 1e-5:
            labels[labels == i] = 0
        else:
            bounding_boxes.append(bounding_box(labels==i))

    img = (labels != 0).astype('uint8') * 255

    bb_centroids = []
    for bb in bounding_boxes:
        bb_centroids.append(((bb[0] + bb[2]) // 2, (bb[1] + bb[3]) // 2))

    bb_centroids = np.array(bb_centroids)
    dist_matrix = distance.cdist(bb_centroids, bb_centroids)
    closest = np.where(dist_matrix < 20)

    to_merge = []

    for i in range(closest[0].shape[0]):
        source = closest[0][i]
        target = closest[1][i]
        if source != target and source < target:
            to_merge.append((source, target))

    bboxes = merge_bboxes(bounding_boxes, to_merge)

    filtered_bb = []
    for bb in bounding_boxes:
        h = bb[3] - bb[1]
        w = bb[2] - bb[0]
        aspect_ratio = w / h
        if aspect_ratio > 0.1:
            filtered_bb.append(bb)
        else:
            img[bb[1]-5:bb[3]+5, bb[0]-5:bb[2]+5] = 0
            labels[bb[1]-5:bb[3]+5, bb[0]-5:bb[2]+5] = 0


    unique = []
    [unique.append(item) for item in filtered_bb if item not in unique]

    return unique

def get_center(bbox):
    return np.array([(bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2])

def draw_bbs(img, bbs):
    if len(img.shape) == 3 and img.shape[2] == 3:
        color = (0, 255, 0)
    else:
        color = 150

    for bb in bbs:
        cv2.rectangle(img, bb[0:2], bb[2:4], color)

def expandBbox(bbox, amount):
    return bbox[0] - amount, bbox[1] - amount, bbox[2] + amount, bbox[3] + amount


def isOverlapping(A, B):
    x1min, y1min, x1max, y1max = A
    x2min, y2min, x2max, y2max = B
    return x1min <= x2max and x2min <= x1max and y1min <= y2max and y2min <= y1max

def _resize_pad_to_square_keep_aspect_ratio(img, size, pad_color=0):
    h, w = img.shape[:2]
    sh, sw = size

    if h > sh or w > sw:
        interp = cv2.INTER_AREA
    else:
        interp = cv2.INTER_CUBIC

    aspect = w / h
    if aspect > 1:
        new_w = sw
        new_h = np.round(new_w / aspect).astype(int)
        pad_vert = (sh - new_h) / 2
        pad_top, pad_bot = np.floor(pad_vert).astype(int), np.ceil(pad_vert).astype(int)
        pad_left, pad_right = 0, 0
    elif aspect < 1:
        new_h = sh
        new_w = np.round(new_h * aspect).astype(int)
        pad_horz = (sw - new_w) / 2
        pad_left, pad_right = np.floor(pad_horz).astype(int), np.ceil(pad_horz).astype(int)
        pad_top, pad_bot = 0, 0
    else:
        new_h, new_w = sh, sw
        pad_left, pad_right, pad_top, pad_bot = 0, 0, 0, 0

    if len(img.shape) is 3 and not isinstance(pad_color, (list, tuple, np.ndarray)):
        pad_color = [pad_color] * 3

    scaled_img = cv2.resize(img, (new_w, new_h), interpolation=interp)
    scaled_img = cv2.copyMakeBorder(
        scaled_img, pad_top, pad_bot, pad_left, pad_right,
        borderType=cv2.BORDER_CONSTANT, value=pad_color
    )

    return scaled_img, new_w, new_h, pad_left, pad_top

def extractPatches(img, bboxes):
    patches = [img[bbox[1]:bbox[3], bbox[0]:bbox[2]] for bbox in bboxes]
    return patches

def normalizePatch(patch):
    patches = _resize_pad_to_square_keep_aspect_ratio(patch, (28, 28))[0]
    return patches

def normalizePatches(patches):
    patches = [normalizePatch(patch) for patch in patches]
    patches = [(p > 100).astype('uint8') * 255 for p in patches]
    return patches

def printPatches(patches):
    f, ax = plt.subplots(1, len(patches))
    for i in range(len(patches)):
        ax[i].imshow(patches[i], cmap='gray')
    plt.show()

def printPatchesAll(a, b):
    f, ax = plt.subplots(2, len(a))
    for i in range(len(a)):
        ax[0,i].imshow(a[i], cmap='gray')
    for i in range(len(b)):
        ax[1,i].imshow(b[i], cmap='jet')
    plt.show(block=False)

def draw_expression(frame, expression):
    cv2.putText(frame, "Expression: " + expression, (100, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

def draw_positions(frame, positions):
    if len(positions) > 0:
        cv2.circle(frame, positions[0], 1, (255, 0, 0), 10)
        for i in range(1, len(positions)):
            cv2.circle(frame, positions[i], 1, (255, 0, 0), 10)
            cv2.line(frame, positions[i-1], positions[i], (255, 255, 0))

