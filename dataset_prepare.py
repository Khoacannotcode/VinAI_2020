from utils import *


# os.environ["WANDB_API_KEY"] = '8f435998b1a6f9a4e59bfaef1deed81c1362a97d'
# os.environ["WANDB_MODE"] = "dryrun"


size_df = pd.read_csv(TRAIN_META_PATH)
size_df.columns = ['image_id', 'h', 'w']

train_df = pd.read_csv(TRAIN_PATH)
train_df = train_df.merge(size_df, on='image_id', how='left')
train_df[['x_min', 'y_min']] = train_df[['x_min', 'y_min']].fillna(0)
train_df[['x_max', 'y_max']] = train_df[['x_max', 'y_max']].fillna(1)

# print(train_df.tail())

# Analyze file

logger.info(f"Train have {train_df['class_name'].nunique()-1} types of thoracic abnormalities")
logger.info(f"Train have {train_df['rad_id'].nunique()} radiologists")
logger.info(f"Train have {train_df['rad_id'].nunique()} radiologists")

train_normal = train_df[train_df['class_name']=='No finding'].reset_index(drop=True)
train_normal['x_min_resize'] = 0
train_normal['y_min_resize'] = 0
train_normal['x_max_resize'] = 1
train_normal['y_max_resize'] = 1

train_abnormal = train_df[train_df['class_name']!='No finding'].reset_index(drop=True)
train_abnormal[['x_min_resize', 'y_min_resize', 'x_max_resize', 'y_max_resize']] = train_abnormal \
.apply(lambda x: label_resize(x[['h', 'w']].values, IMG_SIZE, *x[['x_min', 'y_min', 'x_max', 'y_max']].values),
       axis=1, result_type="expand")
train_abnormal['x_center'] = 0.5*(train_abnormal['x_min_resize'] + train_abnormal['x_max_resize'])
train_abnormal['y_center'] = 0.5*(train_abnormal['y_min_resize'] + train_abnormal['y_max_resize'])
train_abnormal['width'] = train_abnormal['x_max_resize'] - train_abnormal['x_min_resize']
train_abnormal['height'] = train_abnormal['y_max_resize'] - train_abnormal['y_min_resize']
train_abnormal['area'] = train_abnormal.apply(lambda x: (x['x_max_resize']-x['x_min_resize'])*(x['y_max_resize']-x['y_min_resize']), axis=1)
train_abnormal = train_abnormal[~train_abnormal.index.isin(list_remove)].reset_index(drop=True)

print(train_abnormal['class_id'].value_counts())