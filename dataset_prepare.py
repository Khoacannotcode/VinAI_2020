from utils import *
import glob

os.environ["WANDB_API_KEY"] = '8f435998b1a6f9a4e59bfaef1deed81c1362a97d'
os.environ["WANDB_MODE"] = "dryrun"

MAIN_PATH = '/home/VinBigData_ChestXray'
CLASSIFIER_MAIN_PATH = os.path.join(MAIN_PATH, 'efficient_model', 'efficientnet')

TRAIN_PATH = os.path.join(MAIN_PATH, 'train.csv')
SUB_PATH = os.path.join(MAIN_PATH, 'sample_submission.csv')
TRAIN_DICOM_PATH = os.path.join(MAIN_PATH, 'train')
TEST_DICOM_PATH = os.path.join(MAIN_PATH, 'test')

TRAIN_ORIGINAL_PATH = os.path.join(MAIN_PATH, 'train_jpg')
TEST_ORiGINAL_PATH = os.path.join(MAIN_PATH, 'test_jpg')

TRAIN_META_PATH = os.path.join(MAIN_PATH, 'train_meta.csv')
TEST_META_PATH = os.path.join(MAIN_PATH, 'test_meta.csv')

# TEST_CLASS_PATH = '../input/vinbigdata-2class-prediction/2-cls test pred.csv'
MODEL_WEIGHT = os.path.join(CLASSIFIER_MAIN_PATH, 'tf_efficientdet_d7_53-6d1d7a95.pth')

train_dicom_list = glob.glob(f'{TRAIN_DICOM_PATH}/*.dicom')
test_dicom_list = glob.glob(f'{TEST_DICOM_PATH}/*.dicom')

train_list = glob.glob(f'{TRAIN_ORIGINAL_PATH}/*.png')
test_list = glob.glob(f'{TEST_ORiGINAL_PATH}/*.png')
logger.info(f'Train have {len(train_list)} file and test have {len(test_list)}')


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

# print(train_abnormal.tail())