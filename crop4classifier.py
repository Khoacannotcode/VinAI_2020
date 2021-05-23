import pandas as pd
import glob
import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import argparse
from sklearn.model_selection import train_test_split
import numpy as np
from tensorflow.keras.utils import Sequence, to_categorical
import cv2
import keras
import tensorflow as tf
from keras.applications import MobileNetV2
from tensorflow.keras.applications import EfficientNetB0, EfficientNetB4, EfficientNetB7, EfficientNetB5
from keras.layers import Flatten, Dense, Input, Dropout, AveragePooling2D
from keras.models import Model
from keras.optimizers import Adam, SGD, RMSprop
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.preprocessing.image import ImageDataGenerator
from keras.models import model_from_json
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions 


import numpy as np
import matplotlib.gridspec as gridspec
from tqdm import tqdm
import shutil

from tensorflow.compat.v2.keras.utils import multi_gpu_model


def get_path(dir_path, number):
  path = os.path.join(dir_path, number)
  path, _, files = next(os.walk(path))
  list_path = []
  for fn in files:
    tail = fn.split(".")[-1]
    if tail == "gif":
      print("not get ", fn)
    else:
      image_path = os.path.join(path, fn)
      list_path.append(image_path)

  return list_path

def create_folder(PATH):
    if os.path.exists(PATH):
        print(f"{PATH} is exists")
        pass 
    else:
        os.mkdir(PATH)
        print(f"{PATH} is created")

# Augmentation
class DataGenerator(Sequence):
    'Generates data for Keras'
    def __init__(self,
                 all_filenames, 
                 labels, 
                 batch_size, 
                 index2class,
                 input_dim,
                 n_channels,
                 n_classes=2, 
                 normalize=True,
                 zoom_range=[0.8, 1],
                 rotation=15,
                 brightness_range=[0.8, 1],
                 shuffle=True):
        '''
        all_filenames: list toàn bộ các filename
        labels: nhãn của toàn bộ các file
        batch_size: kích thước của 1 batch
        index2class: index của các class
        input_dim: (width, height) đầu vào của ảnh
        n_channels: số lượng channels của ảnh
        n_classes: số lượng các class 
        normalize: có chuẩn hóa ảnh hay không?
        zoom_range: khoảng scale zoom là một khoảng nằm trong [0, 1].
        rotation: độ xoay ảnh.
        brightness_range: Khoảng biến thiên cường độ sáng
        shuffle: có shuffle dữ liệu sau mỗi epoch hay không?
        '''
        self.all_filenames = all_filenames
        self.labels = labels
        self.batch_size = batch_size
        self.index2class = index2class
        self.input_dim = input_dim
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.normalize = normalize
        self.zoom_range = zoom_range
        self.rotation = rotation
        self.brightness_range = brightness_range
        self.on_epoch_end()

    def __len__(self):
        '''
        return:
          Trả về số lượng batch/1 epoch
        '''
        return int(np.floor(len(self.all_filenames) / self.batch_size))

    def __getitem__(self, index):
        '''
        params:
          index: index của batch
        return:
          X, y cho batch thứ index
        '''
        # Lấy ra indexes của batch thứ index
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # List all_filenames trong một batch
        all_filenames_temp = [self.all_filenames[k] for k in indexes]

        # Khởi tạo data
        X, y = self.__data_generation(all_filenames_temp)

        return X, y

    def on_epoch_end(self):
        '''
        Shuffle dữ liệu khi epochs end hoặc start.
        '''
        self.indexes = np.arange(len(self.all_filenames))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, all_filenames_temp):
        '''
        params:
          all_filenames_temp: list các filenames trong 1 batch
        return:
          Trả về giá trị cho một batch.
        '''
        X = np.empty((self.batch_size, *self.input_dim, self.n_channels))
        y = np.empty((self.batch_size), dtype=int)

        # Khởi tạo dữ liệu
        for i, fn in enumerate(all_filenames_temp):
            # Đọc file từ folder name
            img = cv2.imread(fn)
            try:
              img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
              img = cv2.resize(img, self.input_dim)
              img_reshape = img.reshape(-1, 3)
              
              if self.normalize:
                mean = np.mean(img_reshape, axis=0)
                std = np.std(img_reshape, axis=0)
                img = (img-mean)/std

              if self.zoom_range:
                zoom_scale = 1/np.random.uniform(self.zoom_range[0], self.zoom_range[1])
                (h, w, c) = img.shape
                img = cv2.resize(img, (int(h*zoom_scale), int(w*zoom_scale)), interpolation = cv2.INTER_LINEAR)
                (h_rz, w_rz, c) = img.shape
                start_w = np.random.randint(0, w_rz-w) if (w_rz-w) > 0 else 0
                start_h = np.random.randint(0, h_rz-h) if (h_rz-h) > 0 else 0
                # print(start_w, start_h)
                img = img[start_h:(start_h+h), start_w:(start_w+w), :].copy()
              
              if self.rotation:
                (h, w, c) = img.shape
                angle = np.random.uniform(-self.rotation, self.rotation)
                RotMat = cv2.getRotationMatrix2D(center = (w, h), angle=angle, scale=1)
                img = cv2.warpAffine(img, RotMat, (w, h))

              if self.brightness_range:
                scale_bright = np.random.uniform(self.brightness_range[0], self.brightness_range[1])
                img = img*scale_bright
              
              img = cv2.resize(img, (1024, 1024), 3)
              # dataset_task2/data_classify/0/*.jpg
              label = fn.split("/")[-2]
              label = self.index2class[label]
      
              X[i,] = img

              # Lưu class
              y[i] = label

            except:
              pass
            
        return X, y


def create_model(baseModel, number_class, lr=1e-4, decay=1e-4/25, num_gpus=G):
    for layer in baseModel.layers:
          layer.trainable = False
    headModel = baseModel.output
    headModel = AveragePooling2D(pool_size=(3, 3))(headModel)
    headModel = Flatten(name="flatten")(headModel)
    headModel = Dense(1024, activation="relu")(headModel)
    headModel = Dropout(0.5)(headModel)
    headModel = Dense(number_class, activation="softmax")(headModel)
    
    # model = Model(inputs=baseModel.input, outputs=headModel)
    
    # we'll store a copy of the model on *every* GPU and then combine
    # the results from the gradient updates on the CPU
    with tf.device("/cpu:0"):
      model = Model(inputs=baseModel.input, outputs=headModel)

    # make the model parallel
    model = multi_gpu_model(model, gpus=num_gpus)

    # compile model
    optimizer = SGD(lr=lr, decay = decay)

    model.compile(loss="sparse_categorical_crossentropy", optimizer=optimizer,metrics=["accuracy"])    

    return model


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='PyTorch Inference')
  parser.add_argument('-d', '--data', metavar='DIR',
                      help='path to dataset')
  parser.add_argument('--output_dir', metavar='DIR', default='./',
                      help='path to output files')
  parser.add_argument('--model', '-m', metavar='MODEL', default='tf_efficientnet_b7',
                      help='model architecture (default: tf_efficientnet_b7)')
  parser.add_argument('-j', '--workers', default=2, type=int, metavar='N',
                      help='number of data loading workers (default: 2)')
  parser.add_argument('-b', '--batch_size', default=256, type=int,
                      metavar='N', help='mini-batch size (default: 256)')
  parser.add_argument('--img-size', default=None, type=int,
                      metavar='N', help='Input image dimension')
  parser.add_argument('--mean', type=float, nargs='+', default=None, metavar='MEAN',
                      help='Override mean pixel value of dataset')
  parser.add_argument('--std', type=float, nargs='+', default=None, metavar='STD',
                      help='Override std deviation of of dataset')
  parser.add_argument('--interpolation', default='', type=str, metavar='NAME',
                      help='Image resize interpolation type (overrides model)')
  parser.add_argument('--num_classes', type=int, default=1,
                      help='Number classes in dataset')
  parser.add_argument('--list_classes', type=str, default=None, 
                      help='List of name of classes (Example: 1, 2, 3, 4, 5)')
  parser.add_argument('--log-freq', default=10, type=int,
                      metavar='N', help='batch logging frequency (default: 10)')
  parser.add_argument('--checkpoint', default='', type=str, metavar='PATH',
                      help='path to latest checkpoint (default: none)')
  parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                      help='use pre-trained model')
  parser.add_argument('--num_gpu', type=int, default=7,
                      help='Number of GPUS to use')
  parser.add_argument('--no-test-pool', dest='no_test_pool', action='store_true',
                      help='disable test time pool')
  parser.add_argument('--topk', default=5, type=int,
                      metavar='N', help='Top-k to output to CSV')

  args = parser.parse_args()

  labels = []
  image_links = []
  for cls in args.list_classes.split(','):
    class_data = get_path(cls)
    print(f"Length class {cls}: ", len(class_data))
    class_labels = [f'{cls}']*len(class_data)

    labels += class_labels
    image_links += class_data

  # split dataset
  images_train, images_val, y_label_train, y_label_val = train_test_split(image_links, labels, stratify = labels)
  print('images_train len: {}, image_test shape: {}'.format(len(images_train), len(images_val)))

  dict_labels = {}
  for idx, cls in enumerate(args.list_classes.split(',')):
    dict_labels[f'{cls}'] = idx

  train_generator = DataGenerator(
      all_filenames = images_train,
      labels = y_label_train,
      batch_size = 32,
      index2class = dict_labels,
      input_dim = (1024, 1024),
      n_channels = 3,
      n_classes = 4,
      normalize = False,
      zoom_range = [0.5, 1],
      rotation = False,
      brightness_range=[0.8, 1],
      shuffle = True
  )

  val_generator = DataGenerator(
      all_filenames = images_val,
      labels = y_label_val,
      batch_size = 16,
      index2class = dict_labels,
      input_dim = (1024, 1024),
      n_channels = 3,
      n_classes = 4,
      normalize = False,
      zoom_range = [0.5, 1],
      rotation = False,
      brightness_range =[0.8, 1],
      shuffle = False
  )
  # set GPU
  G = 2
  # disable eager execution
  tf.compat.v1.disable_eager_execution()
  print("[INFO] training with {} GPUs...".format(G))

  model = EfficientNetB7(input_shape=(1024, 1024, 3),
                      include_top = False,
                      weights='imagenet')
    
  # config params
  INIT_LR = 2e-4
  EPOCHS = 100
  DECAY = 1e-2
  model = create_model(model, 4, lr=INIT_LR, decay=DECAY)
  print(model.summary())

  #load weights
  model.load_weights("/home/VinBigData_ChestXray/data_classify/B6_100epoch.h5")

  # Start training
  my_checkpointer = [
                  EarlyStopping(monitor='val_loss', patience=5, verbose=0),
                  ModelCheckpoint(filepath="/home/VinBigData_ChestXray/data_classify/B6_100epoch.h5", verbose=2, save_weights_only=True, mode='max', save_best_only=True)
                  ]

  history = model.fit(train_generator,
                    steps_per_epoch=20, 
                     validation_data= val_generator, 
                    validation_steps=10, 
                     epochs=EPOCHS, 
                     callbacks=my_checkpointer)
  
  print("Saved model {}".format("B7_100epoch.h5"))

  model_json = model.to_json()
  with open("/home/VinBigData_ChestXray/data_classify/EfficientNetB7_classification.json", "w") as json_file:
    json_file.write(model_json)

  # Save and plot results
  print(history.history.keys())
  # summarize history for accuracy
  plt.plot(history.history['accuracy'])
  plt.plot(history.history['val_accuracy'])
  plt.title('model accuracy')
  plt.ylabel('accuracy')
  plt.xlabel('epoch')
  plt.legend(['train', 'test'], loc='upper left')
  plt.savefig("/home/VinBigData_ChestXray/data_classify/accuracy.png")
  # summarize history for loss
  plt.plot(history.history['loss'])
  plt.plot(history.history['val_loss'])
  plt.title('model loss')
  plt.ylabel('loss')
  plt.xlabel('epoch')
  plt.legend(['train', 'test'], loc='upper left')
  plt.savefig("/home/VinBigData_ChestXray/data_classify/loss.png")

  # Evaluate model 

  print('Preprocessing and predicting data')
  test_images = [idx for idx in glob.glob('/home/VinBigData_ChestXray/data_classify/4_10_11_12_single_class_ver2_stage1_test_cropped/*.png')]

  classes_lst = []
  class2id = {
    0: '4',
    1: '10',
    2: '11',
    3: '12'
  }
  RESULTS_PATH = '/home/VinBigData_ChestXray/data_classify/results_4_10_11_12_single_class_ver2_stage1_test_cropped/'


  create_folder(RESULTS_PATH)
  create_folder(os.path.join(RESULTS_PATH, '4'))
  create_folder(os.path.join(RESULTS_PATH, '10'))
  create_folder(os.path.join(RESULTS_PATH, '11'))
  create_folder(os.path.join(RESULTS_PATH, '12'))


  for image_idx in test_images:
    img = image.load_img(image_idx, target_size=(1024, 1024))
    img_array = image.img_to_array(img)
    img_batch = np.expand_dims(img_array, axis=0)
    img_preprocessed = preprocess_input(img_batch)
    prediction = model.predict(img_preprocessed)
    classes = class2id[np.argmax(prediction[0])]
    print('Saving {} as {}'.format(image_idx.split('/')[-1], classes + '_' + image_idx.split('/')[-1]))
    shutil.copyfile(image_idx, os.path.join(RESULTS_PATH, classes, classes + '_' + image_idx.split('/')[-1]))

  print('Length of test images', len(classes_lst))