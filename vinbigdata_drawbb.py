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
import seaborn as sns
import gc

BB_PATH = '/home/mmlab/github/vinbigdata-xray/DrawBoundingBox'
TRAIN_PATH = '/home/mmlab/github/vinbigdata-xray/VinBigData_Xray_DownSize/train/train'
df_train = pd.read_csv('/home/mmlab/github/vinbigdata-xray/VinBigData_Xray_DownSize/train_downsampled.csv')
int_2_str = {i:df_train[df_train["class_id"]==i].iloc[0]["class_name"] for i in range(15)}

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


def check_exist_folder(PATH):
    # Check folder is exist
    try:
        os.mkdir(PATH)
        print("{} is created".format(PATH.split("/")[-1]))
    except:
        if os.path.exists(PATH):
            print("{} is exists".format(PATH.split("/")[-1]))

def draw_bboxes(img, tl, br, rgb, label="", label_location="tl", opacity=0.1, line_thickness=0):
    rect = np.uint8(np.ones((br[1]-tl[1], br[0]-tl[0], 3))*rgb)
    sub_combo = cv2.addWeighted(img[tl[1]:br[1],tl[0]:br[0],:], 1-opacity, rect, opacity, 1.0)    
    img[tl[1]:br[1],tl[0]:br[0],:] = sub_combo

    if line_thickness>0:
        img = cv2.rectangle(img, tuple(tl), tuple(br), rgb, line_thickness)
        
    if label:
        # DEFAULTS
        FONT = cv2.FONT_HERSHEY_SIMPLEX
        FONT_SCALE = 1.666
        FONT_THICKNESS = 3
        FONT_LINE_TYPE = cv2.LINE_AA
        
        if type(label)==str:
            LABEL = label.upper().replace(" ", "_")
        else:
            LABEL = f"CLASS_{label:02}"
        
        text_width, text_height = cv2.getTextSize(LABEL, FONT, FONT_SCALE, FONT_THICKNESS)[0]
        
        label_origin = {"tl":tl, "br":br, "tr":(br[0],tl[1]), "bl":(tl[0],br[1])}[label_location]
        label_offset = {
            "tl":np.array([0, -10]), "br":np.array([-text_width, text_height+10]), 
            "tr":np.array([-text_width, -10]), "bl":np.array([0, text_height+10])
        }[label_location]
        img = cv2.putText(img, LABEL, tuple(label_origin+label_offset), 
                          FONT, FONT_SCALE, rgb, FONT_THICKNESS, FONT_LINE_TYPE)
    
    return img

def plot_image(img, title="", figsize=(12,8), cmap=None):
    """ Function to plot an image to save a bit of time """
    plt.figure(figsize=figsize)
    
    if cmap:
        plt.imshow(img, cmap=cmap)
    else:
        img
        plt.imshow(img)
        
    plt.title(title, fontweight="bold")
    plt.axis(False)
    plt.show()

class TrainData():
    def __init__(self, df, train_dir, cmap="Spectral"):
        # Initialize
        self.df = df
        self.train_dir = train_dir
        
        # Visualization
        self.cmap = cmap
        self.pal = [tuple([int(x) for x in np.array(c)*(255,255,255)]) for c in sns.color_palette(cmap, 15)]
        self.pal.pop(8)
        
        # Store df components in individual numpy arrays for easy access based on index
        tmp_numpy = self.df.to_numpy()
        image_ids = tmp_numpy[0]
        class_ids = tmp_numpy[1]
        rad_ids = tmp_numpy[2]
        bboxes = tmp_numpy[3:]
        
        self.img_annotations = self.get_annotations(get_all=True)
        
        # Clean-Up
        del tmp_numpy; gc.collect();
        
        
    def get_annotations(self, get_all=False, image_ids=None, class_ids=None, rad_ids=None, index=None):
        """ TBD 
        
        Args:
            get_all (bool, optional): TBD
            image_ids (list of strs, optional): TBD
            class_ids (list of ints, optional): TBD
            rad_ids (list of strs, optional): TBD
            index (int, optional):
        
        Returns:
        
        
        """
        if not get_all and image_ids is None and class_ids is None and rad_ids is None and index is None:
            raise ValueError("Expected one of the following arguments to be passed:" \
                             "\n\t\t– `get_all`, `image_id`, `class_id`, `rad_id`, or `index`")
        # Initialize
        tmp_df = self.df.copy()
        
        if not get_all:
            if image_ids is not None:
                tmp_df = tmp_df[tmp_df.image_id.isin(image_ids)]
            if class_ids is not None:
                tmp_df = tmp_df[tmp_df.class_id.isin(class_ids)]
            if rad_ids is not None:
                tmp_df = tmp_df[tmp_df.rad_id.isin(rad_ids)]
            if index is not None:
                tmp_df = tmp_df.iloc[index]
            
        annotations = {image_id:[] for image_id in tmp_df.image_id.to_list()}
        for row in tmp_df.to_numpy():
            
            # Update annotations dictionary
            annotations[row[0]].append(dict(
                img_path=os.path.join(self.train_dir, row[0]+".dicom"),
                image_id=row[0],
                class_id=int(row[1]),
                rad_id=int(row[2][1:]),
            ))
            
            # Catch to convert float array to integer array
            if row[1]==14:
                annotations[row[0]][-1]["bbox"]=row[3:]
            else:
                annotations[row[0]][-1]["bbox"]=row[3:].astype(np.int32)
        return annotations
    
    def get_annotated_image(self, image_id, annots=None, plot=False, plot_size=(18,25), plot_title=""):
        if annots is None:
            annots = self.img_annotations.copy()
        
        if type(annots) != list:
            image_annots = annots[image_id]
        else:
            image_annots = annots
            
        img = cv2.cvtColor(cv2.imread(image_annots[0]["img_path"]),cv2.COLOR_GRAY2RGB)
        for ann in image_annots:
            if ann["class_id"] != 14:
                img = draw_bboxes(img, 
                                ann["bbox"][:2], ann["bbox"][-2:], 
                                rgb=self.pal[ann["class_id"]], 
                                label=int_2_str[ann["class_id"]], 
                                opacity=0.01, line_thickness=4)
        if plot:
            plot_image(img, title=plot_title, figsize=plot_size)
        
        return img
    
        
    def plot_classes(self, class_list, n=4, height_multiplier=6, verbose=True):
        annotations = self.get_annotations(class_ids=class_list)
        annotated_imgs = []

        plt.figure(figsize=(20, height_multiplier*n))
        for i, (image_id, annots) in enumerate(annotations.items()):
            if i >= n:
                break
            if verbose:
                print(f".", end="")
            plt.subplot(n//2,2,i+1)
            plt.imshow(self.get_annotated_image(image_id, annots))
            plt.axis(False)
            plt.title(f"Image ID – {image_id}")
        plt.tight_layout(rect=[0, 0.03, 1, 0.97])
        plt.show()


if __name__ == "__main__":
    train_data = TrainData(df_train, TRAIN_PATH)
    train_data.plot_classes(class_list=[7,], n=1, verbose=False)
