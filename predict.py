from utils import *
# from dataset_prepare import *
from postprocessing import *


class Predict_process(object):
    def __init__(self, device=device, config=PredictConfig):
        super(Predict_process, self).__init__()
        self.device = device
        self.config = config
        
    def load_image(self, image_path): # :, transforms):
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # image = transforms(image=image)['image']
        return image
        
    def classifier_image(self, images):
        model = EfficientnetCus(model=self.config.model_classifier_use, num_class=14,
                                model_weight=self.config.weight_classifier, is_train=False).to(self.device)
        model.eval()
        with torch.no_grad():
            outputs = model(images.to(device))
        return outputs
    
    def label_process(self, detect_result, iou_thresh, iou_thresh11):
        assert detect_result != ''
        x_center, y_center = detect_result[1::6], detect_result[2::6]
        w_center, h_center = detect_result[3::6], detect_result[4::6]
        detect_result[1::6] = [i-0.5*j for i, j in zip(x_center, w_center)]
        detect_result[2::6] = [i-0.5*j for i, j in zip(y_center, h_center)]
        detect_result[3::6] = [i+0.5*j for i, j in zip(x_center, w_center)]
        detect_result[4::6] = [i+0.5*j for i, j in zip(y_center, h_center)]
        list_new = []
        
        for label_values in np.unique(detect_result[::6]):
            list_values = np.array([detect_result[6*idx:6*idx+6] \
                                    for idx, i in enumerate(detect_result[::6]) if i==label_values])
            boxes = list_values[:, 1:5].tolist()
            scores = list_values[:, 5].tolist()
            labels = list_values[:, 0].tolist()
            if label_values in [2, 11]:
                boxes, scores, labels = nms([boxes], [scores], [labels], weights=None, iou_thr=iou_thresh11)
            else:
                boxes, scores, labels = nms([boxes], [scores], [labels], weights=None, iou_thr=iou_thresh)
            
            for box in list_values:
                if box[-1] in scores:
                    list_new.extend(box)
        return list_new
        
    def read_label(self, label_path):
        with open(label_path, 'r+') as file:
            detect_result = file.read()
        if detect_result != '':
            detect_result = list(map(float, re.split(r'[\n ]', detect_result)[:-1]))
            detect_result = self.label_process(detect_result, self.config.iou_thresh, self.config.iou_thresh11)
            detect_result = [int(i) if idx%6==0 else self.config.img_size[0]*i if idx%6<5 else i 
                             for idx, i in enumerate(detect_result)]
        return detect_result
        
    def fit(self, df, folder_image, result_txt, use_classifier=True):
        transforms = aug('test')
        all_results = []
        for images_id, images in tqdm(df.iterrows(), total=len(df), leave=False):
            if use_classifier:
                image_path = os.path.join(folder_image, images.image_id+'.png')
                image = self.load_image(image_path) #, transforms)
                class_labels = self.classifier_image(image)
            else:
                class_labels = torch.tensor([1])
            label_path = os.path.join(result_txt, images.image_id+'.txt')
            if os.path.isfile(label_path):
                detect_result = self.read_label(label_path)
            else:
                detect_result = ''
            result_one_image = []
            if detect_result != '':
                img_size = [images.h, images.w]
                list_label = []
                for box_id in range(len(detect_result)//6)[::-1]:
                    label, *box, score = detect_result[6*box_id:6*box_id+6]
                    if class_labels.item()>=self.config.classification_thresh:
                        if (score > self.config.score_last) and \
                        not(label in [0, 3] and label in list_label) and \
                        not(label==11 and score < self.config.score_11) and \
                        not(label==9 and score < self.config.score_9):
                            list_label.append(label)
                            box = label_resize(self.config.img_size, img_size, *box)
                            result_one_image.append(int(label))
                            result_one_image.append(np.round(score, 3))
                            result_one_image.extend([int(i) for i in box])
                    else:
                        if score > self.config.score_last2 and \
                        not(label in [0, 3] and label in list_label) and \
                        not(label==11 and score < self.config.score_11) and \
                        not(label==9 and score < self.config.score_9):
                            list_label.append(label)
                            box = label_resize(self.config.img_size, img_size, *box)
                            result_one_image.append(int(label))
                            result_one_image.append(np.round(score, 3))
                            result_one_image.extend([int(i) for i in box])
            if len(result_one_image)==0:
                all_results.append('14 1 0 0 1 1')
            else:
                result_str = ' '.join(map(str, result_one_image)) + \
                f' 14 {class_labels.item():.3f} 0 0 1 1'
                all_results.append(result_str)
        df['PredictionString'] = all_results
        df = df.drop(['h', 'w'], 1)
        
        return df

if __name__ == '__main__':
    size_df = pd.read_csv(TEST_META_PATH)
    size_df.columns = ['image_id', 'h', 'w']

    sub_df = pd.read_csv(SUB_PATH)
    sub_df = sub_df.merge(size_df, on='image_id', how='left')
    print(sub_df.head())

    predict_pr = Predict_process(config=PredictConfig)
    submission_df = predict_pr.fit(sub_df, TEST_ORIGINAL_PATH, '/home/appuser/vinbigdata_utils/yolov5/vinbigdata/exp/labels')
    # submission_df.to_csv(os.path.join(MAIN_PATH,'submission_efficientnet.csv'), index=False)
    # print(submission_df.head(60))
