import pandas as pd
import numpy as np

import plotly.express as px
import plotly.graph_objects as go

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from tqdm import tqdm
import argparse
import json


class VinBigDataEval:
    def __init__(self, true_df):
        
        self.true_df = true_df

        self.image_ids = true_df["image_id"].unique()
        self.annotations = {
            "type": "instances",
            "images": self.__gen_images(self.image_ids),
            "categories": self.__gen_categories(self.true_df),
            "annotations": self.__gen_annotations(self.true_df, self.image_ids)
        }
        
        self.predictions = {
            "images": self.annotations["images"].copy(),
            "categories": self.annotations["categories"].copy(),
            "annotations": None
        }
        
    def __gen_categories(self, df):
        print("Generating category data...")
        
        if "class_name" not in df.columns:
            df["class_name"] = df["class_id"]
        
        cats = df[["class_name", "class_id"]]
        cats = cats.drop_duplicates().sort_values(by='class_id').values
        
        results = []
        
        for cat in tqdm(cats):
            results.append({
                "id": cat[1],
                "name": cat[0],
                "supercategory": "none",
            })
            
        return results

    def __gen_images(self, image_ids):
        print("Generating image data...")
        results = []

        for idx, image_id in tqdm(enumerate(image_ids)):

            # Add image identification.
            results.append({
                "id": idx,
            })
            
        return results
    
    def __gen_annotations(self, df, image_ids):
        print("Generating annotation data...")
        k = 0
        results = []
        
        for idx, image_id in tqdm(enumerate(image_ids)):

            # Add image annotations
            for i, row in df[df["image_id"] == image_id].iterrows():

                results.append({
                    "id": k,
                    "image_id": idx,
                    "category_id": row["class_id"],
                    "bbox": np.array([
                        row["x_min"],
                        row["y_min"],
                        row["x_max"],
                        row["y_max"]]
                    ),
                    "segmentation": [],
                    "ignore": 0,
                    "area":(row["x_max"] - row["x_min"]) * (row["y_max"] - row["y_min"]),
                    "iscrowd": 0,
                })

                k += 1
                
        return results

    def __decode_prediction_string(self, pred_str):
        data = list(map(float, pred_str.split(" ")))
        data = np.array(data)

        return data.reshape(-1, 6)    
    
    def __gen_predictions(self, df, image_ids):
        print("Generating prediction data...")
        k = 0
        results = []
        
        for i, row in tqdm(df.iterrows()):
            
            image_id = row["image_id"]
            preds = self.__decode_prediction_string(row["PredictionString"])

            for j, pred in enumerate(preds):

                results.append({
                    "id": k,
                    "image_id": int(np.where(image_ids == image_id)[0]),
                    "category_id": int(pred[0]),
                    "bbox": np.array([
                        pred[2], pred[3], pred[4], pred[5]
                    ]),
                    "segmentation": [],
                    "ignore": 0,
                    "area": (pred[4] - pred[2]) * (pred[5] - pred[3]),
                    "iscrowd": 0,
                    "score": pred[1]
                })

                k += 1
                
        return results
                
    def evaluate(self, pred_df, n_imgs = -1, iou_thres=0.4):
        """Evaluating your results
        
        Arguments:
            pred_df: pd.DataFrame your predicted results in the
                     competition output format.

            n_imgs:  int Number of images use for calculating the
                     result.All of the images if `n_imgs` <= 0
                     
        Returns:
            COCOEval object
        """
        
        if pred_df is not None:
            self.predictions["annotations"] = self.__gen_predictions(pred_df, self.image_ids)

        coco_ds = COCO()
        coco_ds.dataset = self.annotations
        coco_ds.createIndex()
        
        coco_dt = COCO()
        coco_dt.dataset = self.predictions
        coco_dt.createIndex()
        
        imgIds=sorted(coco_ds.getImgIds())
        
        if n_imgs > 0:
            imgIds = np.random.choice(imgIds, n_imgs)

        cocoEval = COCOeval(coco_ds, coco_dt, 'bbox')
        cocoEval.params.imgIds  = imgIds
        cocoEval.params.useCats = True
        cocoEval.params.iouType = "bbox"
        cocoEval.params.iouThrs = np.array([iou_thres])

        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()
        
        return cocoEval

class VinBigDataEvalJson(VinBigDataEval):
    def __init__(self, json_file):
        self.json_file = json_file

        self.image_ids = [imgs['file_name'].split('.')[0] for imgs in self.json_file['images']]
        self.annotations = {
            "type": "instances",
            "images": [{"id": i} for i in range(len(self.json_file['images']))],
            "categories": self.json_file['categories'],
            "annotations": self.json_file['annotations']
        }
        
        self.predictions = {
            "images": self.annotations["images"].copy(),
            "categories": self.annotations["categories"].copy(),
            "annotations": None
        }
    
    def evaluate(self, json_file_pred, n_imgs = -1, iou_thres=0.4):
        self.predictions["annotations"] = json_file_pred['annotations']

        coco_ds = COCO()
        coco_ds.dataset = self.annotations
        coco_ds.createIndex()
        
        coco_dt = COCO()
        coco_dt.dataset = self.predictions
        coco_dt.createIndex()
        
        imgIds = sorted(coco_ds.getImgIds())
        
        if n_imgs > 0:
            imgIds = np.random.choice(imgIds, n_imgs)

        cocoEval = COCOeval(coco_ds, coco_dt, 'bbox')
        cocoEval.params.imgIds  = imgIds
        cocoEval.params.useCats = True
        cocoEval.params.iouType = "bbox"
        cocoEval.params.iouThrs = np.array([iou_thres])

        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()
        
        return cocoEval


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", help="Directory of ground truth in .csv pascalVoc format or coco format", type=str)
    parser.add_argument("--pred_dir", help="Directory of prediction path", type=str)
    parser.add_argument("--iou_thres", help="IOU threshold", type=float)
    args = parser.parse_args()

    if args.root_dir.endswith(".json"):
        with open(args.root_dir, 'r') as json_file:
            data = json.load(json_file)
        vineval = VinBigDataEvalJson(data)
    else:
        df = pd.read_csv(args.root_dir)
        df.fillna(0, inplace=True)
        df.loc[df["class_id"] == 14, ['x_max', 'y_max']] = 1.0
        df = df.groupby(by=['image_id', 'class_id']).first().reset_index()
        vineval = VinBigDataEval(df)


    if args.pred_dir.endswith(".json"):
        with open(args.pred_dir, 'r') as json_file_pred:
            data_pred = json.load(json_file_pred)
        cocoEvalRes = vineval.evaluate(data_pred, iou_thres=args.iou_thres)
    else:
        pred_df = pd.read_csv(args.pred_dir)
        pred_df = pred_df.drop_duplicates()
        pred_df.reset_index(drop=True, inplace=True)
        cocoEvalRes = vineval.evaluate(pred_df, iou_thres=args.iou_thres)
