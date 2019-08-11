import tensorflow as tf 
import os
import re 


def _activation_summary(x,TOWER_NAME = None): 
  """Helper to create summaries for activations.
  Creates a summary that provides a histogram of activations.
  Creates a summary that measures the sparsity of activations.
  Args:
    x: Tensor
  Returns:
    nothing
  """
  # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
  # session. This helps the clarity of presentation on tensorboard.
  tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
  tf.summary.histogram(tensor_name + '/activations', x)
  tf.summary.scalar(tensor_name + '/sparsity', tf.nn.zero_fraction(x))


def _variable_on_cpu(name,shape,stddev,initializer=None):
    with tf.device('/cpu:0'):
        
        if initializer == None:
            var = tf.get_variable(name ,shape,initializer=tf.truncated_normal_initializer(stddev=stddev))
        else :
            var = tf.get_variable(name,shape,initializer=initializer)
    return var 


def _variable_with_decay(name,shape ,stddev,wd ):
    var = _variable_on_cpu(name,shape,stddev,initializer=None)
    if wd is not None:
        weight_decay = tf.multiply(tf.nn.l2_loss(var),wd,name="weight_loss")
        tf.add_to_collections("losses",weight_decay)
    return var

def inference (FLAGS,image,Reuse=False):
    print(Reuse)
    with tf.variable_scope("conv1",reuse =Reuse):
        kernel = _variable_with_decay('weights',shape=[5,5,3,64],stddev =5e-2, wd =None)
        conv1 = tf.nn.conv2d(image,kernel,[1,1,1,1],padding='SAME')
        bias =_variable_on_cpu('biases',[64],stddev=0,initializer= tf.constant_initializer(0.1) )
        output = tf.nn.bias_add(conv1,bias)
        output_conv1 = tf. nn.relu(output)
    with tf.variable_scope('pool1',reuse =Reuse):
        pool1 = tf.nn.max_pool(output_conv1,[1,3,3,1],[1,2,2,1],padding='SAME',name='pool_conv1')
        norm1 = tf.nn.lrn(pool1,depth_radius=4,beta = 0.75,name= 'norm1')
    
    with tf.variable_scope("conv2",reuse =Reuse):
        kernel = _variable_with_decay('weights',shape=[5,5,64,64],stddev =5e-2, wd =None)
        conv2 = tf.nn.conv2d(norm1,kernel,[1,1,1,1],padding='SAME')
        bias =_variable_on_cpu('biases',[64],stddev=0,initializer= tf.constant_initializer(0.1) )
        output = tf.nn.bias_add(conv2,bias)
        output_conv2 = tf. nn.relu(output)

    with tf.variable_scope('pool2',reuse =Reuse):
        norm2 = tf.nn.lrn(output_conv2,depth_radius=4,beta = 0.75,name='norm1')
    
        pool2 = tf.nn.max_pool(norm2,[1,3,3,1],[1,2,2,1],padding='SAME',name='poo2_conv2')
    
    with tf.variable_scope("fc1",reuse =Reuse):
        reshape = tf.keras.layers.Flatten()(pool2)
        dim = reshape.get_shape()[1].value
        weight = _variable_with_decay('weights',shape=[dim,384],stddev =0.04, wd =0.004)
      
        bias =_variable_on_cpu('biases',[384],stddev=0,initializer= tf.constant_initializer(0.1) )

        output_fc1 = tf. nn.relu(tf.matmul(reshape,weight)+bias)

        ## we also can utilize the droppot even using the weight decay before then
        out_fc1 =tf.nn.dropout (output_fc1,0.4)
        
    with tf.variable_scope('fc2',reuse =Reuse):
        weight = _variable_with_decay('weights',[384,192],stddev=0.04,wd=0.004)
        biases = _variable_on_cpu('biases',[192],stddev=0,initializer=tf.constant_initializer(0.1))
        output_fc2 = tf.nn.dropout(tf.matmul(out_fc1,weight)+biases,0.4)

    with tf.variable_scope('softmax-linear',reuse =Reuse):
        weight = _variable_with_decay('weights',[192,FLAGS.NUM_CLASS],stddev=0.04,wd= None)
        biases = _variable_on_cpu('biases',[FLAGS.NUM_CLASS],stddev= 0,initializer=tf.constant_initializer(0.1))
        logits = tf.add(tf.matmul(output_fc2,weight),biases,name = "logits")

    return logits


    
    
        