import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
from pathlib import Path
import json
import re
import logging
import gc
import random
import warnings
from tqdm.notebook import tqdm
from PIL import Image
import os
from glob import glob
from joblib import Parallel, delayed
import shutil as sh
from itertools import product
from collections import OrderedDict
import plotly
import plotly.express as px
import plotly.graph_objects as go
import plotly.offline as pyo
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from ensemble_boxes import nms, weighted_boxes_fusion
import torchvision.transforms as transforms
import albumentations as al
from albumentations import ImageOnlyTransform
from albumentations.pytorch import ToTensorV2, ToTensor
from albumentations.core.transforms_interface import DualTransform, ImageOnlyTransform

import cv2
import pydicom
# from IPython.display import display, Image
from sklearn.model_selection import train_test_split, KFold, GroupKFold, StratifiedKFold
import torch
from torch.nn import functional as f
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau, CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader, sampler
import timm
from efficientnet_pytorch import EfficientNet
import glob

pd.options.display.max_columns = None
warnings.filterwarnings('ignore')

logging.basicConfig(format='%(asctime)s +++ %(message)s',
                    datefmt='%d-%m-%y %H:%M:%S', level=logging.INFO)
logger = logging.getLogger()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(device)


SIZE = 1024
IMG_SIZE = (SIZE, SIZE)
ACCULATION = 1
MOSAIC_RATIO = 0.4


MAIN_PATH = '/home/VinBigData_ChestXray'
CLASSIFIER_MAIN_PATH = os.path.join(MAIN_PATH, 'efficient_model', 'efficientnet')

TRAIN_PATH = os.path.join(MAIN_PATH, 'anti_radConflict.csv')
SUB_PATH = os.path.join(MAIN_PATH, 'sample_submission.csv')
TRAIN_DICOM_PATH = os.path.join(MAIN_PATH, 'train')
TEST_DICOM_PATH = os.path.join(MAIN_PATH, 'test')

TRAIN_ORIGINAL_PATH = os.path.join(MAIN_PATH, 'train_jpg')
TEST_ORIGINAL_PATH = os.path.join(MAIN_PATH, 'test_jpg')

TRAIN_META_PATH = os.path.join(MAIN_PATH, 'train_meta.csv')
TEST_META_PATH = os.path.join(MAIN_PATH, 'test_meta.csv')

# TEST_CLASS_PATH = '../input/vinbigdata-2class-prediction/2-cls test pred.csv'
MODEL_WEIGHT = os.path.join(CLASSIFIER_MAIN_PATH, 'tf_efficientdet_d7_53-6d1d7a95.pth')

train_dicom_list = glob.glob(f'{TRAIN_DICOM_PATH}/*.dicom')
test_dicom_list = glob.glob(f'{TEST_DICOM_PATH}/*.dicom')

train_list = glob.glob(f'{TRAIN_ORIGINAL_PATH}/*.png')
test_list = glob.glob(f'{TEST_ORIGINAL_PATH}/*.png')
logger.info(f'Train have {len(train_list)} file and test have {len(test_list)}')


    
class GlobalConfig:
    model_use = 'd5'
    model_weight = None
    img_size = IMG_SIZE
    fold_num = 3
    seed = 89
    num_workers = 0
    batch_size = 8
    n_epochs = 20
    lr = 1e-2
    verbose = 1
    verbose_step = 1
    step_scheduler = False  # do scheduler.step after optimizer.step
    validation_scheduler = True  # do scheduler.step after validation stage loss
    SchedulerClass = torch.optim.lr_scheduler.ReduceLROnPlateau
#     output_path = './save/'
    scheduler_params = dict(
        mode='min', 
        factor=0.2,
        patience=1,
        threshold_mode='abs',
        min_lr=1e-7
    )
    
class PredictConfig:
    img_size = IMG_SIZE
    batch_size = 8
    model_classifier_use = 'b5'
    weight_classifier = None
#     weight_classifier = '../input/x-chest-1024-classifier/model_classifier.pth'

    score_thresh = 0.05
    iou_thresh = 0.5
    iou_thresh2 = 0.1
    iou_thresh11 = 0.0001
    skip_thresh = 0.0001
    sigma = 0.1
    score_0 = 0.385
    score_3 = 0.4
    score_last = 0.0
    score_last2 = 0.95
    score_9 = 0.1
    score_11 = 0.015
    classification_thresh = 0.003751
    
list_remove = [34843, 21125, 647, 18011, 2539, 22373, 12675, 7359, 20642, 5502, 19818, 5832, 28056, 28333, 20758,
               925, 43, 2199, 4610, 21306, 16677, 1768, 17232, 1378, 24949, 30203, 31410, 87, 25318, 92, 31724,
               118, 17687, 12605, 26157, 33875, 7000, 3730, 18776, 13225, 1109, 2161, 33627, 15500, 28633, 28152,
               10114, 10912, 9014,  4427, 25630, 11464, 6419, 22164, 4386, 17557, 15264, 21853, 33142, 32895, 9733,
               33010, 17493, 32128, 28802, 11658, 8841, 29557, 4802, 8591, 778, 9935, 12359, 5210, 7556, 24505, 5664,
               28670, 27820, 19359, 9817, 7800, 32934, 34098, 27931, 16074, 27308, 30645, 31029, 35697, 6199, 27065,
               1771, 14689, 31860, 1975, 29294, 2304, 34018, 23406, 26501, 26011, 2479, 32796, 25836, 3032, 31454,
               32066, 19722, 15997, 6049, 9458, 11005, 23151, 24503, 35411, 18092, 23815, 30742, 33942, 34542, 7655,
               25345, 3750, 17046, 3844, 5958, 4250, 18823, 14898, 22581, 25805, 9651, 33194, 36007, 30160, 24459,
               10838, 16544, 31252, 8053, 28487, 6208, 25244, 8470, 10089, 24813, 14769, 34305, 34047, 23366, 8049,
               13276, 22380, 32797, 32440, 11031, 18304, 33692, 21349, 26333, 34331, 9110, 21092, 34882, 35626, 10203,
               25648, 30754, 29567, 33542, 15146, 26759, 20846, 22493, 33187, 22813, 30219, 14548, 14627, 20494, 28332,
               15930, 31347, 33489, 35005, 34032, 24183, 18643, 18536, 29754, 20380, 29750, 20539, 35791, 27275, 32248]


image_remove = ['9c83d9f88170cd38f7bca54fe27dc48a', 'ac2a615b3861212f9a2ada6acd077fd9',
                'f9f7feefb4bac748ff7ad313e4a78906', 'f89143595274fa6016f6eec550442af9',
                '6c08a98e48ba72aee1b7b62e1f28e6da', 'e7a58f5647d24fc877f9cb3d051792e2',
                '8f98e3e6e86e573a6bd32403086b3707', '43d3137e74ebd344636228e786cb91b0',
                '575b98a9f9824d519937a776bd819cc4', 'ca6c1531a83f8ee89916ed934f8d4847',
                '0c6a7e3c733bd4f4d89443ca16615fc6', 'ae5cec1517ab3e82c5374e4c6219a17d',
                '064023f1ff95962a1eee46b9f05f7309', '27c831fee072b232499541b0aca58d9c',
                '0b98b21145a9425bf3eeea4b0de425e7', '7df5c81873c74ecc40610a1ad4eb2943']




class BaseWheatTTA:
    """ author: @shonenkov """
    image_size = IMG_SIZE

    def augment(self, image):
        raise NotImplementedError
    
    def batch_augment(self, images):
        raise NotImplementedError
    
    def deaugment_boxes(self, boxes):
        raise NotImplementedError

class TTAHorizontalFlip(BaseWheatTTA):
    """ author: @shonenkov """

    def augment(self, image):
        return image.flip(1)
    
    def batch_augment(self, images):
        return images.flip(2)
    
    def deaugment_boxes(self, boxes):
        boxes[:, [1,3]] = self.image_size - boxes[:, [3,1]]
        return boxes

class TTAVerticalFlip(BaseWheatTTA):
    """ author: @shonenkov """
    
    def augment(self, image):
        return image.flip(2)
    
    def batch_augment(self, images):
        return images.flip(3)
    
    def deaugment_boxes(self, boxes):
        boxes[:, [0,2]] = self.image_size - boxes[:, [2,0]]
        return boxes
    
class TTARotate90(BaseWheatTTA):
    """ author: @shonenkov """
    
    def augment(self, image):
        return torch.rot90(image, 1, (1, 2))

    def batch_augment(self, images):
        return torch.rot90(images, 1, (2, 3))
    
    def deaugment_boxes(self, boxes):
        res_boxes = boxes.copy()
        res_boxes[:, [0,2]] = self.image_size - boxes[:, [1,3]]
        res_boxes[:, [1,3]] = boxes[:, [2,0]]
        return res_boxes

class TTACompose(BaseWheatTTA):
    """ author: @shonenkov """
    def __init__(self, transforms):
        self.transforms = transforms
        
    def augment(self, image):
        for transform in self.transforms:
            image = transform.augment(image)
        return image
    
    def batch_augment(self, images):
        for transform in self.transforms:
            images = transform.batch_augment(images)
        return images
    
    def prepare_boxes(self, boxes):
        result_boxes = boxes.copy()
        result_boxes[:,0] = np.min(boxes[:, [0,2]], axis=1)
        result_boxes[:,2] = np.max(boxes[:, [0,2]], axis=1)
        result_boxes[:,1] = np.min(boxes[:, [1,3]], axis=1)
        result_boxes[:,3] = np.max(boxes[:, [1,3]], axis=1)
        return result_boxes
    
    def deaugment_boxes(self, boxes):
        for transform in self.transforms[::-1]:
            boxes = transform.deaugment_boxes(boxes)
        return self.prepare_boxes(boxes)
    
def Visualize_class(df, feature, title):
    num_image = df[feature].value_counts().rename_axis(feature).reset_index(name='num_image')
    fig = px.bar(num_image[::-1], x='num_image', y=feature, orientation='h', color='num_image')
    fig.update_layout(
    title={
        'text': title,
        'y':0.95,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'})
    fig.show()
    
    
def img_size(path):
    information = pydicom.dcmread(path)
    h, w = information.Rows, information.Columns
    return (h, w)


def label_resize(org_size, img_size, *bbox):
    x0, y0, x1, y1 = bbox
    x0_new = int(np.round(x0*img_size[1]/org_size[1]))
    y0_new = int(np.round(y0*img_size[0]/org_size[0]))
    x1_new = int(np.round(x1*img_size[1]/org_size[1]))
    y1_new = int(np.round(y1*img_size[0]/org_size[0]))
    return x0_new, y0_new, x1_new, y1_new


def list_color(class_list):
    dict_color = dict()
    for classid in class_list:
        dict_color[classid] = [i/256 for i in random.sample(range(256), 3)]
    
    return dict_color


def split_df(df):
    kf = MultilabelStratifiedKFold(n_splits=GlobalConfig.fold_num,
                                   shuffle=True, random_state=GlobalConfig.seed)
    df['id'] = df.index
    annot_pivot = pd.pivot_table(df, index='image_id', columns='class_id',
                                 values='id', fill_value=0, aggfunc='count') \
    .reset_index().rename_axis(None, axis=1)
    for fold, (train_idx, val_idx) in enumerate(kf.split(annot_pivot,
                                                         annot_pivot.iloc[:, 1:train_abnormal['class_id'].nunique()])):
        annot_pivot[f'fold_{fold}'] = 0
        annot_pivot.loc[val_idx, f'fold_{fold}'] = 1
    return annot_pivot
    
    
def display_image(df, list_image, num_image=1, is_dicom_file=True):
    
    dict_color = list_color(range(15))
    list_abnormal = [i for i in df['class_name'].unique() if i!='No finding']
    for abnormal in list_abnormal:
        abnormal_df = df[df['class_name']==abnormal].reset_index(drop=True)
        abnormal_random = np.random.choice(abnormal_df['image_id'].unique(), num_image)
        for abnormal_img in abnormal_random:
            images = abnormal_df[abnormal_df['image_id']==abnormal_img].reset_index(drop=True)
            fig, ax = plt.subplots(1, figsize=(15, 15))
            img_path = [i for i in list_image if abnormal_img in i][0]
            if is_dicom_file:
                information = pydicom.dcmread(img_path)
                img = information.pixel_array
            else:
                img = cv2.imread(img_path)
            ax.imshow(img, plt.cm.bone)
            for idx, image in images.iterrows():
                bbox = [image.x_min, image.y_min, image.x_max, image.y_max]
                if is_dicom_file:
                    x_min, y_min, x_max, y_max = bbox
                else:
                    org_size = image[['h', 'w']].values
                    x_min, y_min, x_max, y_max = label_resize(org_size, IMG_SIZE, *bbox)
                class_name, class_id = image.class_name, image.class_id
                rect = patches.Rectangle((x_min, y_min), x_max-x_min, y_max-y_min,
                                         linewidth=1, edgecolor=dict_color[class_id], facecolor='none')
                ax.add_patch(rect)
                plt.text(x_min, y_min, class_name, fontsize=15, color='red')

            plt.title(abnormal_img) 
            plt.show()
            
def display_image_test(df, size_df, list_image, num_image=3):
    
    dict_color = list_color(range(15))
    image_row_random = np.random.choice(len(df), num_image, replace=(len(df)<num_image))
    for image_idx in image_row_random:
        image_id, pred = df.loc[image_idx, 'image_id'], df.loc[image_idx, 'PredictionString']
        org_size = size_df[size_df['image_id']==image_id][['h', 'w']].values[0].tolist()
        fig, ax = plt.subplots(1, figsize=(15, 15))
        img_path = [i for i in list_image if image_id in i][0]
        img = cv2.imread(img_path)
        ax.imshow(img, plt.cm.bone)
        if pred != '14 1 0 0 1 1':
            list_pred = pred.split(' ')
            for box_idx in range(len(list_pred)//6):
                bbox = map(int, list_pred[6*box_idx+2:6*box_idx+6])
                x_min, y_min, x_max, y_max = label_resize(org_size, IMG_SIZE, *bbox)
                class_name, score = int(list_pred[6*box_idx]), float(list_pred[6*box_idx+1])
                rect = patches.Rectangle((x_min, y_min), x_max-x_min, y_max-y_min,
                                         linewidth=1, edgecolor=dict_color[class_name], facecolor='none')
                ax.add_patch(rect)
                plt.text(x_min, y_min, f'{class_name}: {score}', fontsize=15, color='red')            

        plt.title(image_id) 
        plt.show()
        
def ensemble_multibox(boxes, scores, labels, iou_thr, sigma,
                      skip_box_thr, weights=None, method='wbf'):
    if method=='nms':
        boxes, scores, labels = nms(boxes, scores, labels,
                                    weights=weights,
                                    iou_thr=iou_thr)
    elif method=='soft_nms':
        boxes, scores, labels = soft_nms(boxes, scores, labels,
                                         weights=weights,
                                         sigma=sigma,
                                         iou_thr=iou_thr,
                                         thresh=skip_box_thr)
    elif method=='nms_weight':
        boxes, scores, labels = non_maximum_weighted(boxes, scores, labels,
                                                     weights=weights,
                                                     iou_thr=iou_thr,
                                                     skip_box_thr=skip_box_thr)
    elif method=='wbf':
        boxes, scores, labels = weighted_boxes_fusion(boxes, scores, labels,
                                                      weights=weights,
                                                      iou_thr=iou_thr,
                                                      skip_box_thr=skip_box_thr)
    
    return boxes, scores, labels