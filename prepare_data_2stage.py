import pandas as pd
from tqdm import tqdm
from shutil import copyfile
import imagesize
import time
import os
from pathlib import Path
import argparse
from icecream import ic

def my_error():
    error = '⚠ 👨‍💻 Please contact with quandzkosoai to fix this issuse'
    return error

class CreateDataset(object):
    def __init__(self, white_list, dataset_name, bb_type, class_options, save_dir="/dataset/", raw_folder='train'):
        self.white_list = white_list
        self.dataset_name = dataset_name
        
        self.original_dataset = "/dataset/vinbigdata-chest-xray-resized-png-1024x1024/"
        self.raw_folder = os.path.join(self.original_dataset, raw_folder)

        self.save_dir = os.path.join(save_dir, dataset_name)
        Path(os.path.join(save_dir, dataset_name)).mkdir(parents=True, exist_ok=True)
        self.train_folder = os.path.join(self.save_dir, self.dataset_name + "_train")
        self.val_folder = os.path.join(self.save_dir, self.dataset_name + "_val")

        self.df_train = pd.read_csv('/dataset/train_wbf_anti_conflict_ver2.csv')
        self.df_train_meta = pd.read_csv('/dataset/train_meta.csv')

        self.bb_type = bb_type
        self.class_options = class_options
    
    def Step1(self):
        Path(self.train_folder).mkdir(parents=True, exist_ok=True)
        white_file = 0
        white_file_list = []

        read_img_time = 0
        ic("whitelist: ", self.white_list)
        for file in tqdm(os.listdir(self.raw_folder), desc="Step 1"):
            if file[0] != ".": # Ignore temp file  
                df_find = self.df_train[(self.df_train.image_id == file[:-4])]

                # Xét ảnh có trong whitelist hay không
                is_valid = True
                cls_list = []
                for index, row in df_find.iterrows():
                    cls_list.append(row[1])
                is_valid = any(item in cls_list for item in self.white_list)
                if is_valid == False:
                    continue
            
                if len(df_find) > 0:
                    # Kiểm tra để dễ debug
                    for index, row in df_find.iterrows():
                        cls_list.append(row[1])
                    if not any(item in cls_list for item in self.white_list):
                        raise Exception(my_error())

                    white_file += 1
                    # Copy file này qua bên train_whitelist
                    src_whitelist = os.path.join(self.raw_folder, file)
                    dst_whitelist = os.path.join(self.train_folder, file)
                    copyfile(src_whitelist, dst_whitelist)
                    
                    t = time.time()
                    img_width, img_height = imagesize.get(src_whitelist)
                    image_size = img_width
                    read_img_time += (time.time() - t)
                    if(image_size !=1024):
                        ic("⚠: ERROR IMG SIZE")
                        raise Exception(my_error())
                    # image_size = 1024
                    meta_frame = self.df_train_meta[(self.df_train_meta.image_id == file[:-4])].values
                    O_W, O_H = meta_frame[0][2], meta_frame[0][1]

                    labels = []
                    for index, row in df_find.iterrows():
                        if row[1] not in self.white_list:
                            continue
                        box_width = row[4] - row[2]
                        box_height = row[5] - row[3]
                        box_center_x = (row[4] + row[2]) / 2
                        box_center_y = (row[5] + row[3]) / 2        
                        
                        
                        if self.bb_type.lower() == 'yolo': 
                            box_center_x = box_center_x/O_W
                            box_center_y = box_center_y/O_H
                            box_width = box_width/O_W
                            box_height = box_height/O_H

                            labels.append([row[1], box_center_x, box_center_y, box_width, box_height])
                        elif self.bb_type.lower() == 'coco': 
                            # Chuyển đám trên sang xywh hệ pixel
                            box_center_x = box_center_x * 1024
                            box_center_y = box_center_y * 1024
                            box_width = box_width * 1024
                            box_height = box_height * 1024

                            # Sau đó tính ra xmin ymin xmax ymax
                            xmin = box_center_x - (box_width // 2)
                            ymin = box_center_y - (box_height // 2)
                            xmax = box_center_x + (box_width // 2)
                            ymax = box_center_y + (box_height // 2)

                            labels.append([row[1], xmin, ymin, xmax, ymax])
                        
                        else:
                            raise TypeError("Only support Coco and YOLO format.")
                        
                    txt_file = file[:-4] + ".txt"
                    with open(os.path.join(self.train_folder, txt_file),'w') as f:
                        for label in labels:
                            if self.class_options == 1:
                                f.write('{} {} {} {} {}\n'.format(str(0),label[1],label[2],label[3],label[4]))
                            elif self.class_options == 2:
                                f.write('{} {} {} {} {}\n'.format(label[0],label[1],label[2],label[3],label[4]))
                            
                white_file_list.append(file)


        ic("Done")
        ic(f"White_list has {white_file} pics")
        ic("Check length of white_file_list: ", len(white_file_list))
        ic("Time to read all pics: ", read_img_time)

        return white_file, white_file_list

    def Step2(self, white_file):
        nokosu = 0
        nokosu_list = []

        
        Path(self.val_folder).mkdir(parents=True, exist_ok=True)


        for file in tqdm(os.listdir(self.raw_folder), desc="Step 2"):
            if file[0] != ".": # Ignore temp file
                df_find = self.df_train[(self.df_train.image_id == file[:-4])]

                # Nếu ảnh có cls_id thuộc whitelist hoặc bằng 14 thì invalid
                is_invalid = False
                for index, row in df_find.iterrows():
                    if (row[1] in self.white_list) or (row[1] == 14):
                        is_invalid = True
                if is_invalid == True:
                    continue
            
                if len(df_find) > 0:
                    for index, row in df_find.iterrows():
                        if (row[1] in self.white_list) or (row[1] == 14):
                            raise Exception(my_error())

                    nokosu += 1
                    if nokosu > (white_file//2):
                        break

                    src2 = os.path.join(self.raw_folder, file)
                    dst2 = os.path.join(self.val_folder, file)
                    copyfile(src2, dst2)
                
                nokosu_list.append(file)

        ic("Done")
        ic("Number of pics conatins nokosu: ", nokosu)
        ic("Check nokosu_list's length: ", len(nokosu_list))

        return nokosu, nokosu_list

    def Step3(self, white_file, nokosu):
        cls_14 = 0
        list_cls_14 = []

        blacklist = [ x for x in range(14)]

        for file in tqdm(os.listdir(self.raw_folder), desc="Step 3"):
            if file[0] != ".": # Ignore temp file  
                
                df_find = self.df_train[(self.df_train.image_id == file[:-4])]

                # Nếu ảnh có cls_id trong blacklist thì invalid
                is_invalid = False
                for index, row in df_find.iterrows():
                    if row[1] in blacklist:
                        is_invalid = True
                if is_invalid == True:
                    continue
            
                if len(df_find) > 0:
                    for index, row in df_find.iterrows():
                        if row[1] in blacklist:
                            raise Exception(my_error())

                    cls_14 += 1
                    if cls_14 > (white_file - nokosu):
                        break

                    src1 = os.path.join(self.raw_folder, file)
                    dst1 = os.path.join(self.val_folder, file)
                    copyfile(src1, dst1)
                list_cls_14.append(file)

        ic("Number of pics depends on cls 14: ", cls_14)
        ic("Check list_cls_14's length: ", len(list_cls_14))
        ic("Done")        

        return cls_14, list_cls_14

    def running(self):
        white_file, white_file_list = self.Step1()
        nokosu, nokosu_list = self.Step2(white_file)
        _, list_cls_14 = self.Step3(white_file, nokosu)

        ic("Check white_file_list")
        t = time.time()
        for img in white_file_list:
            if img in nokosu_list or img in list_cls_14:
                raise Exception(my_error())

        ic("Time execution: ", time.time()-t)

        ic("Check nokosu_list")
        t = time.time()
        for img in nokosu_list:
            if img in white_file_list or img in list_cls_14:
                raise Exception(my_error())

        ic("Time execution: ", time.time()-t)

        ic("Check list_cls_14")
        t = time.time()
        for img in list_cls_14:
            if img in nokosu_list or img in white_file_list:
                raise Exception(my_error())

        ic("Time execution: ", time.time()-t)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Prepare dataset two stage')
    parser.add_argument("--whitelist", metavar="-w", type=str, required=True, default=None, help="White list of classes (Example: 1,2,3,4).")
    parser.add_argument("--dataset_name", metavar="-d", type=str, required=True, default=None, help="Name of dataset.")
    parser.add_argument("--bb_type", metavar="-b", type=str, default=None, required=True, help="Convert bounding box to COCO or YOLO format. (Example: COCO)")
    parser.add_argument("--class_options", metavar="-c", type=int, required=True, default=0, help="Choose 1 to generate single class data and choose 2 to generate multi class data.")

    args = parser.parse_args()

    if not args.bb_type or not args.dataset_name or not args.whitelist:
        raise ValueError("Type python prepare_data_2stage.py --help to more information.")
    
    if args.class_options not in [1, 2]:
        raise ValueError("Please choose 1 or 2 to generate data.")
    
    dataset_singleclass = CreateDataset(white_list=[int(x) for x in args.whitelist.split(",")], dataset_name=args.dataset_name, bb_type=args.bb_type, class_options=args.class_options)
    dataset_singleclass.running()
