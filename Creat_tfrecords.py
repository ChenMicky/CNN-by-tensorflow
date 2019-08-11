import tensorflow as tf 
import glob as gb
from cv2 import cv2 
import numpy as np
from random import shuffle
import matplotlib.pyplot as plt


shuffle_data= True 
classes = {'good','rust','spot','defect','cavity','injured','others'} ## the classification files names 
class write_to_TFrecord:

    def __init__ (self,FLAGS,image_type=None):

        if(image_type == None):
            self._type = '\*.bmp'
        else:
            self._type=image_type
        self._train_path =FLAGS.image_dir
        ##self._label_path =FLAGS.label_dir
        self._img_heigh =FLAGS.image_h
        self._img_width =FLAGS.image_w
        self._save_path =FLAGS.save_tfrecord

    def read_file(self):
        img_list =[]
        
        for x in gb.glob(self._train_path+self._type):
            img_list.append(x)
        
        if shuffle_data:
            
            shuffle(img_list)
            
        return img_list 
    
    def _Load_image_label(self,Img):   
        img =cv2.imread(Img)###turn to one dimension

        img = cv2.resize(img, (self._img_width,self._img_heigh), interpolation=cv2.INTER_CUBIC)
       
        img =np.array(img)
        img = img.tobytes()
        
        ##img =np.array2string(img)## revise
        # label = cv2.imread(Label,cv2.IMREAD_GRAYSCALE)
        # print(label.size)
        # label = cv2.resize(label, (self._img_width,self._img_heigh), interpolation=cv2.INTER_CUBIC)

        # label = label.tostring()
        ##label =np.array2string(label)
        return img
    def transfer_to_TFrecord(self):

        ## feature dict depends on the data to change bytes ,strings,int .etc
        ## also  the "name" in feature group could be  changed and added other attributes that u wanna 
        writer = tf.python_io.TFRecordWriter(self._save_path+'/train_data.tfrecords')  
        for index ,name in enumerate(classes):
            
            
            path = self._train_path+name+"/"
            for img in gb.glob(path+self._type):
                print(index)
                imgs = self._Load_image_label(img)
               
                feature = {'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.compat.as_bytes(imgs)])),
                'label':tf.train.Feature(int64_list=tf.train.Int64List(value=[index]))}##define the key and value ,it likes dictinary 

                example = tf.train.Example(features=tf.train.Features(feature=feature))  ##build it 
                writer.write(example.SerializeToString()) ## serialize it 
        writer.close()

class Read_TFrecord:
    def __init__(self,FLAGS,tfrecord_path):
        self._tfrecord_path = tfrecord_path
        self._img_heigh =FLAGS.image_h
        self._img_width =FLAGS.image_w
        self._Resize = 100
        self._batch_size =FLAGS.batch_size
    def Dataset_read(self):##first way 

        dataset = tf.data.TFRecordDataset(self._tfrecord_path)
        dataset =dataset.map(self._parse_function)##map func :the _parse_function as delivery parameter

        dataset =dataset.repeat().shuffle(self._batch_size)
        dataset=dataset.batch(self._batch_size)
    
        
        iterator = dataset.make_one_shot_iterator()
        next_element= iterator.get_next()
       
        
        return next_element
        
    def _parse_function(self,example_proto):
        features = {  
            'image': tf.FixedLenFeature([], tf.string),  
            'label': tf.FixedLenFeature([], tf.int64)  
            }  
        parsed_features=tf.parse_single_example(example_proto,features)
        image = tf.decode_raw(parsed_features['image'], tf.uint8)
        
        image = tf.reshape(image,[self._img_heigh,self._img_width,3])
        image = tf.image.resize_images(image,[self._Resize,self._Resize],method=3)
        Labels= parsed_features['label']
        
        label = tf.reshape(Labels,[1])
        #label = tf.image.resize_images(label,[self._Resize,self._Resize],method=3)
        image =tf.cast(image, tf.float32)
       
        return image,label