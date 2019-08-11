import tensorflow as tf 
import os
from train import*
from inference import *
import Creat_tfrecords as ct
import matplotlib.pyplot as plt
import numpy as np 
import time
from datetime import datetime
FLAGS = tf.app.flags.FLAGS

NUM_EXAMPLES_PER_EPOCH_FOR_TEST = 10


tf.app.flags.DEFINE_string('testing', 'C:\\Users\\Lab434\\Desktop\\test\\TensorFlow\\Logs\\model.ckpt-999', """ checkpoint file """)
tf.app.flags.DEFINE_string('finetune', '', """ finetune checkpoint file """)
tf.app.flags.DEFINE_integer('batch_size', "3", """ batch_size """)
tf.app.flags.DEFINE_float('learning_rate', "1e-3", """ initial lr """)
tf.app.flags.DEFINE_string('save_tfrecord', "C:\\Users\\Micky\\Desktop\\Original", """ dir to store TFRecord""")
tf.app.flags.DEFINE_string('TFrecord_dir', "C:/Users/Micky/Desktop/Original/train_data.tfrecords", """ path to tfrecord""")
tf.app.flags.DEFINE_string('log_dir', "C:\\Users\\Lab434\\Desktop\\test\\TensorFlow\\Logs", """ dir to store ckpt """)
tf.app.flags.DEFINE_string('image_dir', "C:/Users/Micky/Desktop/Original/", """ path to CamVid image """)
tf.app.flags.DEFINE_string('test_dir', "C:/Users/Lab434/Desktop/test/", """ path to CamVid test image """)
tf.app.flags.DEFINE_string('val_dir', "C:\\Users\\Micky\\Desktop\\test\\test_data.tfrecords", """ path to CamVid val image """)
tf.app.flags.DEFINE_integer('max_steps', "10000", """ max_steps """)
tf.app.flags.DEFINE_integer('image_h', "280", """ image height """)
tf.app.flags.DEFINE_integer('image_w', "280", """ image width """)
tf.app.flags.DEFINE_integer('image_c', "3", """ image channel (RGB) """)
tf.app.flags.DEFINE_integer('NUM_CLASS', "7", """ total class number """)
tf.app.flags.DEFINE_string('save_model', "C:\\Users\\Micky\\Desktop\\Original\\logs\\", """save .ckpt model path""")
is_finetune =False
MOVING_AVERAGE_DECAY = 0.999


os.environ["TF_CPP_MIN_LOG_LEVEL"]='1' #check

os.environ["TF_CPP_MIN_LOG_LEVEL"]='2' #check
os.environ["TF_CPP_MIN_LOG_LEVEL"]='3' #check

def M_training():
    startstep = 0 if not is_finetune else int(FLAGS.finetune.split('-')[-1])

    CR =ct.Read_TFrecord(FLAGS,FLAGS.TFrecord_dir)##train data
    next_element = CR.Dataset_read()
    
    CR_Test =ct.Read_TFrecord(FLAGS,FLAGS.val_dir)## test data
    next_element_test = CR_Test.Dataset_read() 
    
    ##test = next_element[0].shape
    global_step = tf.Variable(0,trainable=False)
    
    logits = inference(FLAGS,next_element[0])

    total_loss, cross_entropy_mean,accurancy = loss (logits,next_element[1],reuse=False)
    
    train_op =train(FLAGS,total_loss,global_step)
    ##############################################################################
   

    #######save
    saver = tf.train.Saver(tf.global_variables())

    summary_op = tf.summary.merge_all()
    
    with tf.Session() as sess:
        if (is_finetune == True):
          
          saver.restore(sess,FLAGS. finetune )
        else:
          init = tf.global_variables_initializer()
          sess.run(init)
        
       
        
        
        
        for step in range(startstep,startstep+FLAGS.max_steps):  
            

            start_time = time.time() 
            _,loss_value =sess.run([train_op,total_loss])   
            duration = time.time()-start_time
           
            sess.run(total_loss)
            #print(loss_value)

            assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

            if step%10 ==0:
                num_examples_per_step = FLAGS.batch_size
                examples_per_sec = num_examples_per_step / duration
                sec_per_batch = float(duration)
                format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                      'sec/batch)')
                print (format_str % (datetime.now(), step, loss_value,
                               examples_per_sec, sec_per_batch))

            if step % 1000 ==0:
                print("validating...")
                total_val_toss =0.0
                sess.run(next_element_test)
                TEST_ITER = NUM_EXAMPLES_PER_EPOCH_FOR_TEST / FLAGS.batch_size
                
                logits_test = inference(FLAGS,next_element_test[0],Reuse=True)

                _val_total_loss, _val_cross_entropy_mean,_val_accurancy = loss (logits_test,next_element_test[1],reuse=True)
                
                for test_step in range(int(TEST_ITER)):
                    total_val_toss+= _val_total_loss
                    sess.run(_val_total_loss)
                print("val_loss:",total_val_toss/TEST_ITER)
                
                
                if step ==(FLAGS.max_steps-1):
                    checkpoint_path = os.path.join(FLAGS.save_model, 'model.ckpt')
                    saver.save(sess, checkpoint_path, global_step=step)
                   
def M_test(FLAGS):

    CR_Test =ct.Read_TFrecord(FLAGS,FLAGS.test_dir)## test data
    next_element_test = CR_Test.Dataset_read() 
    
    ##test = next_element[0].shape
    
    logits = inference(FLAGS,next_element_test[0])

    total_loss, cross_entropy_mean,accurancy = loss (logits,next_element_test[1],reuse=True)
    ####################################

    variable_averages = tf.train.ExponentialMovingAverage(
                      MOVING_AVERAGE_DECAY)
    variables_to_restore = variable_averages.variables_to_restore()

    saver = tf.train.Saver(variables_to_restore)

    

    with tf.Session() as sess:
    # Load checkpoint
        saver.restore(sess, test_ckpt )
        for i in range(FLAGS.batch_size):
            total_loss = 0
            total_acc = 0
            sess.run(next_element_test)
            loss , acc = sess.run(total_loss,accurancy)
            total_loss +=loss
            total_acc+= acc 

        print("avg_loss: ",total_loss/FLAGS.batch_size)
        print("avg_accuracy:",total_acc/FLAGS.batch_size)
                     
                
    
   

def M_transferToTFrecord():
    Ct = ct.write_to_TFrecord(FLAGS,image_type=None)
    Ct.transfer_to_TFrecord()

def main(self):
    
    M_training()
    
        
        

        
if __name__ =='__main__':
    tf.app.run()
""" var1 = tf.Variable([1.0], dtype=tf.float32, name='v1') 
var2 = tf.Variable([2.0], dtype=tf.float32, name='v2') 
addop = tf.add(var1, var2, name='add') 
initop = tf.global_variables_initializer() 
model_path = 'A:/test_img_TFRecords/model.pb' 


test = tf.get_variable_scope().original_name_scope
print(test)
with tf.Session() as sess: 
    
    sess.run(initop)
    print(sess.run(test))
    sess.run(addop)
    graph_def = tf.get_default_graph().as_graph_def() 

    output_graph_def = tf.graph_util.convert_variables_to_constants(sess, sess.graph_def, output_node_names=['add']) 
# 將匯出模型存入檔案 
    with tf.gfile.GFile(model_path, 'wb') as f: 
        f.write(output_graph_def.SerializeToString())  """