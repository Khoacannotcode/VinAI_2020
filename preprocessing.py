from dataset_prepare import *

def Preprocess_wbf(df, size=SIZE, iou_thr=0.5, skip_box_thr=0.0001):
    list_image = []
    list_boxes = []
    list_cls = []
    list_h, list_w = [], []
    new_df = pd.DataFrame()
    for image_id in tqdm(df['image_id'].unique(), leave=False):
        image_df = df[df['image_id']==image_id].reset_index(drop=True)
        h, w = image_df.loc[0, ['h', 'w']].values
        boxes = image_df[['x_min_resize', 'y_min_resize',
                          'x_max_resize', 'y_max_resize']].values.tolist()
        boxes = [[j/(size-1) for j in i] for i in boxes]
        scores = [1.0]*len(boxes)
        labels = [float(i) for i in image_df['class_id'].values]
        boxes, scores, labels = weighted_boxes_fusion([boxes], [scores], [labels],
                                                      weights=None,
                                                      iou_thr=iou_thr,
                                                      skip_box_thr=skip_box_thr)
        list_image.extend([image_id]*len(boxes))
        list_h.extend([h]*len(boxes))
        list_w.extend([w]*len(boxes))
        list_boxes.extend(boxes)
        list_cls.extend(labels.tolist())

    list_boxes = [[int(j*(size-1)) for j in i] for i in list_boxes]
    new_df['image_id'] = list_image
    new_df['class_id'] = list_cls
    new_df['h'] = list_h
    new_df['w'] = list_w
    new_df['x_min_resize'], new_df['y_min_resize'], \
    new_df['x_max_resize'], new_df['y_max_resize'] = np.transpose(list_boxes)
    new_df['x_center'] = 0.5*(new_df['x_min_resize'] + new_df['x_max_resize'])
    new_df['y_center'] = 0.5*(new_df['y_min_resize'] + new_df['y_max_resize'])
    new_df['width'] = new_df['x_max_resize'] - new_df['x_min_resize']
    new_df['height'] = new_df['y_max_resize'] - new_df['y_min_resize']
    new_df['area'] = new_df.apply(lambda x: (x['x_max_resize']-x['x_min_resize'])\
                                  *(x['y_max_resize']-x['y_min_resize']), axis=1)
    return new_df

def split_df(df):
    kf = MultilabelStratifiedKFold(n_splits=GlobalConfig.fold_num, shuffle=True, random_state=GlobalConfig.seed)
    df['id'] = df.index
    annot_pivot = pd.pivot_table(df, index=['image_id'], columns=['class_id'],
                                 values='id', fill_value=0, aggfunc='count') \
    .reset_index().rename_axis(None, axis=1)
    for fold, (train_idx, val_idx) in enumerate(kf.split(annot_pivot,
                                                         annot_pivot.iloc[:, 1:(1+df['class_id'].nunique())])):
        annot_pivot[f'fold_{fold}'] = 0
        annot_pivot.loc[val_idx, f'fold_{fold}'] = 1
    return annot_pivot

def create_file(df, split_df, train_file, train_folder, fold):
    
    os.makedirs(f'{train_file}/labels/train/', exist_ok=True)
    os.makedirs(f'{train_file}/images/train/', exist_ok=True)
    os.makedirs(f'{train_file}/labels/val/', exist_ok=True)
    os.makedirs(f'{train_file}/images/val/', exist_ok=True)
    
    list_image_train = split_df[split_df[f'fold_{fold}']==0]['image_id']    
    train_df = df[df['image_id'].isin(list_image_train)].reset_index(drop=True)
    val_df = df[~df['image_id'].isin(list_image_train)].reset_index(drop=True)
    
    for train_img in tqdm(train_df.image_id.unique()):
        with open(f'{train_file}/labels/train/{train_img}.txt', 'w+') as f:
            row = train_df[train_df['image_id']==train_img]\
            [['class_id', 'x_center', 'y_center', 'width', 'height']].values
            row[:, 1:] /= SIZE
            row = row.astype('str')
            for box in range(len(row)):
                text = ' '.join(row[box])
                f.write(text)
                f.write('\n')
        sh.copy(f'{train_folder}/{train_img}.png', 
                f'{train_file}/images/train/{train_img}.png')
        
    for val_img in tqdm(val_df.image_id.unique()):
        with open(f'{train_file}/labels/val/{val_img}.txt', 'w+') as f:
            row = val_df[val_df['image_id']==val_img]\
            [['class_id', 'x_center', 'y_center', 'width', 'height']].values
            row[:, 1:] /= SIZE
            row = row.astype('str')
            for box in range(len(row)):
                text = ' '.join(row[box])
                f.write(text)
                f.write('\n')
        sh.copy(f'{train_folder}/{val_img}.png', 
                f'{train_file}/images/val/{val_img}.png')

if __name__ == '__main__':
    size_df = pd.read_csv(TRAIN_META_PATH)
    size_df.columns = ['image_id', 'h', 'w']

    fold_csv = split_df(train_abnormal)
    fold_csv = fold_csv.merge(size_df, on='image_id', how='left')
    print(fold_csv.head(10))
    train_abnormal = Preprocess_wbf(train_abnormal)
    print(train_abnormal.head(10))

    create_file(train_abnormal, fold_csv, '/home/VinBigData_ChestXray/yolo_data/yolov5', TRAIN_ORIGINAL_PATH, 3)
    gc.collect()