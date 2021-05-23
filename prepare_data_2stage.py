import pandas as pd
import cv2
import pickle
from tqdm import tqdm
import glob
from shutil import copyfile
import imagesize
import time
import os
from pathlib import Path
import argparse

def my_error():
    error = '⚠ Hệ thống nhận thấy bất thường trong quá trình kiểm tra\n👨‍💻 Vui lòng thông báo quản trị viên quandzkosoai nếu xuất hiện lỗi này'
    return error

class SingeClassDataset(object):
    def __init__(self, white_list, dataset_name, save_dir, raw_folder='train'):
        self.white_list = white_list
        self.dataset_name = dataset_name
        self.save_dir = save_dir
        self.raw_folder = raw_folder

        self.original_dataset = "/dataset/VinBigData_ChestXray/vinbigdata-1024-image-dataset/"
        self.df_train = pd.read('/dataset/VinBigData_ChestXray/train_wbf_anti_conflict_ver2.csv')
        self.df_train_meta = pd.read('/dataset/VinBigData_ChestXray/train_meta.csv')
    
    def Step1(self):
        print("Step 1...")
        train_whitelist = self.dataset_name + "_train"
        Path(train_whitelist).mkdir(parents=True, exist_ok=True)
        white_file = 0
        white_file_list = []

        read_img_time = 0
        print("whitelist: ", self.white_list)
        for file in os.listdir(self.raw_folder):
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
                    dst_whitelist = os.path.join(train_whitelist, file)
                    copyfile(src_whitelist, dst_whitelist)
                    
                    t = time.time()
                    img_width, img_height = imagesize.get(src_whitelist)
                    image_size = img_width
                    read_img_time += (time.time() - t)
                    if(image_size !=1024):
                        print("⚠: ERROR IMG SIZE")
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

                        box_center_x = box_center_x/O_W
                        box_center_y = box_center_y/O_H
                        box_width = box_width/O_W
                        box_height = box_height/O_H

                        # # Chuyển đám trên sang xywh hệ pixel
                        # box_center_x = box_center_x * 1024
                        # box_center_y = box_center_y * 1024
                        # box_width = box_width * 1024
                        # box_height = box_height * 1024

                        # # Sau đó tính ra xmin ymin xmax ymax
                        # xmin = box_center_x - (box_width // 2)
                        # ymin = box_center_y - (box_height // 2)
                        # xmax = box_center_x + (box_width // 2)
                        # ymax = box_center_y + (box_height // 2)


                        labels.append([row[1], box_center_x, box_center_y, box_width, box_height])

                    txt_file = file[:-4] + ".txt"
                    with open(os.path.join(train_whitelist, txt_file),'w') as f:
                        for label in labels:
                            f.write('{} {} {} {} {}\n'.format(str(0),label[1],label[2],label[3],label[4]))
                            # f.write('{} {} {} {} {}\n'.format(label[0],label[1],label[2],label[3],label[4]))
                white_file_list.append(file)


        print("Done")
        print("Số ảnh có white_file là: ", white_file)
        print("Check độ dài của white_file_list: ", len(white_file_list))
        print("Thời gian đọc ảnh:", read_img_time)

        return white_file, white_file_list

    def Step2(self, white_file):
        print("Step 2...")

        nokosu = 0
        nokosu_list = []

        val_folder = self.dataset_name + "_val"
        Path(val_folder).mkdir(parents=True, exist_ok=True)


        for file in os.listdir(self.raw_folder):
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
                    dst2 = os.path.join(val_folder, file)
                    copyfile(src2, dst2)
                
                nokosu_list.append(file)

        print("Done")
        print("Số ảnh có nokosu là: ", nokosu)
        print("Check độ dài của nokosu_list: ", len(nokosu_list))

        return nokosu, nokosu_list

    def Step3(self, white_file, nokosu):
        print("Step 3...")

        cls_14 = 0
        list_cls_14 = []

        blacklist = [ x for x in range(14)]

        for file in os.listdir(self.raw_folder):
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
                    dst1 = os.path.join(val_folder, file)
                    copyfile(src1, dst1)
                list_cls_14.append(file)

        print("Số ảnh thuộc cls_14 = ", cls_14)
        print("Check độ dài của list_cls_14: ", len(list_cls_14))
        print("Done")        

        return cls_14, list_cls_14

    def running(self):
        white_file, white_file_list = self.Step1()
        nokosu, nokosu_list = self.Step2(white_file)
        _, list_cls_14 = self.Step3(white_file, nokosu)

        print("Kiểm tra white_file_list")
        t = time.time()
        for img in white_file_list:
            if img in nokosu_list or img in list_cls_14:
                raise Exception(my_error())

        print("Thời gian thực thi: ", time.time()-t)

        print("Kiểm tra nokosu_list")
        t = time.time()
        for img in nokosu_list:
            if img in white_file_list or img in list_cls_14:
                raise Exception(my_error())

        print("Thời gian thực thi: ", time.time()-t)

        print("Kiểm tra list_cls_14")
        t = time.time()
        for img in list_cls_14:
            if img in nokosu_list or img in white_file_list:
                raise Exception(my_error())

        print("Thời gian thực thi: ", time.time()-t)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Prepare dataset single lcass')
    parser.add_argument()
    parser.add_argument()
    parser.add_argument()
    parser.add_argument()

    args = parser.parse_args()
    
    dataset = SingeClassDataset()