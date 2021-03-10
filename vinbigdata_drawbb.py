import os
import numpy as np 
import pandas as pd 
import cv2
import torch
import matplotlib.pyplot as plt
import albumentations as A
from matplotlib.patches import Rectangle
from tqdm import tqdm
import matplotlib.colors as mcolors

BB_PATH = '/home/mmlab/github/vinbigdata-xray/DrawBoundingBox'
TRAIN_PATH = '/home/mmlab/github/vinbigdata-xray/VinBigData_Xray_DownSize/train/train'
df_train = pd.read_csv('/home/mmlab/github/vinbigdata-xray/VinBigData_Xray_DownSize/train_downsampled.csv')

def check_exist_folder(PATH):
    # Check folder is exist
    try:
        os.mkdir(PATH)
        print("{} is created".format(PATH.split("/")[-1]))
    except:
        if os.path.exists(PATH):
            print("{} is exists".format(PATH.split("/")[-1]))

def fusing_boxes(dataframe, class_id):
    # filter on class_id
    filtered_dataframe = dataframe.loc[dataframe.class_id == class_id,
                                       ['image_id','x_min','y_min','x_max','y_max']]
    # aggregate on image_id to average radiologists's estimations
    return filtered_dataframe.groupby(['image_id']).mean()

def get_rectangle_parameter(dataframe, index):
    
    "Adapt coordinates of bounding box for patch.Rectangle function"
    
    x_min = dataframe.loc[index, 'x_min']
    y_min = dataframe.loc[index, 'y_min']
    x_max = dataframe.loc[index, 'x_max']
    y_max = dataframe.loc[index, 'y_max']
    
    anchor_point = (x_min, y_min)
    height = y_max - y_min
    width = x_max - x_min
    
    return anchor_point, height, width

def select_imageid_from_each(dataframe):
    
    "For each class, returns 9 indexes and image paths"
    
    # Initialize dictionaries
    class_id_index_examples, class_id_image_examples = {}, {}
    image_ids_train_dataframe = list(df_train.image_id)
    class_id_value_counts = df_train["class_id"].value_counts().sort_index()

    # Loop over different classes
    for class_id in range(len(class_id_value_counts.keys())):
        fusing_boxes_dataframe = fusing_boxes(dataframe, class_id)
        # image_id
        fusing_box_indexes = fusing_boxes_dataframe.index
        # Infer indexes
        class_id_index_examples[str(class_id)] = [image_ids_train_dataframe.index(fusing_box_indexes[cid]) for cid in range(fusing_boxes_dataframe.shape[0])]
        # Infer image paths
        class_id_image_examples[str(class_id)] = fusing_box_indexes
        
    return class_id_index_examples, class_id_image_examples

class_id_index_examples, class_id_image_examples = select_imageid_from_each(df_train)

def draw_boundingbox(class_id, graph_indexes):
    # Get files
    files_index = class_id_index_examples[str(class_id)]
    files_list = class_id_image_examples[str(class_id)]
    
    # Create folder by class id
    check_exist_folder(os.path.join(BB_PATH, class_id))

    # Color mapping to edge color of rectangle
    color = {
        '0': 'r',
        '1': 'orangered',
        '2': 'yellow',
        '3': 'chartreuse',
        '4': 'orange',
        '5': 'lime',
        '6': 'aquamarine',
        '7': 'magenta',
        '8': 'cyan',
        '9': 'dodgerblue',
        '10': 'lightcoral',
        '11': 'deepskyblue',
        '12': 'deeppink',
        '13': 'gold'
    }

    # Draw bounding box
    print('Draw bounding box on {} images'.format(len(graph_indexes)))
    fig, ax = plt.subplots()
    for graph_index in tqdm(graph_indexes):
        
        full_filename = files_list[graph_index] + '.jpg'
        img = plt.imread(os.path.join(TRAIN_PATH,
                                  full_filename))
        
        ax.axis('off') 
        ax.imshow(img, cmap=plt.get_cmap('gray'))
                  
        if str(class_id) != '14':
            # Add rectangle
            anchor_point, height, width = get_rectangle_parameter(df_train, 
                                                                  files_index[graph_index])
            rect = Rectangle(anchor_point, 
                                     height, 
                                     width, 
                                     edgecolor=color[str(class_id)], 
                                     facecolor="none")
            ax.add_patch(rect)
        
        plt.savefig(os.path.join(BB_PATH,class_id,full_filename) + '.jpg',dpi=300,transparent=True)
        plt.cla()
    print(f'Saved images in {class_id}')


if __name__ == "__main__":
    for i in range(len(class_id_index_examples)):
        # if len(os.listdir(BB_PATH + '/{}'.format(i))) == len(class_id_index_examples[str(i)]):
        #     continue
        # else:
        draw_boundingbox(str(i),np.arange(len(class_id_index_examples[str(i)])))