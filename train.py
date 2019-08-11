import tensorflow as tf 

NUM_EXAMPLES_PER_EPOXH_FOR_TRAIN = 10000
NUM_EPOCH_PER_DECAY = 350
INITIAL_LEARNING_RATE = 0.1
LEARNING_RATE_DECAY_FACTOR =0.1
MOVING_AVERAGE_DECAY = 0.999

def _losses(label,logits,reuse):
    if reuse ==True:
        _val_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels= label,logits= logits,name= 'cross_entropy')

        _val_cross_entropy_mean =tf.reduce_mean(_val_cross_entropy,name = 'cross_entropy_mean')
    
        tf.add_to_collection('losses', _val_cross_entropy_mean)

        return tf.add_n(tf.get_collection('losses'),name='total_loss'),_val_cross_entropy_mean
    elif reuse ==False:
        cross_entropy1 = tf.nn.softmax_cross_entropy_with_logits(labels= label,logits= logits,name= 'cross_entropy')

        cross_entropy1_mean =tf.reduce_mean(cross_entropy1,name = 'cross_entropy_mean')
    
        tf.add_to_collection('losses', cross_entropy1_mean)
    
        train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy1_mean)
        return tf.add_n(tf.get_collection('losses'),name='total_loss'),cross_entropy1_mean
   
    
def loss(logits,labels,reuse=False):
    labels = tf.cast(labels,tf.float32)
    labelss =tf.cast(labels,tf.int64)
    
    t_labelsK = tf.one_hot(labelss,depth = 7)##depth means how many numbers of classes 
    t_labels =tf.reshape(t_labelsK,(-1,7))
    
   
    
    """ cross_entropy1 = tf.nn.softmax_cross_entropy_with_logits(labels= t_labels,logits= logits,name= 'cross_entropy')

    cross_entropy_mean =tf.reduce_mean(cross_entropy1,name = 'cross_entropy_mean')
    
    tf.add_to_collection('losses', cross_entropy_mean)
   
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy_mean) """
    
    loss,cross_entropy_mean = _losses(t_labels,logits,reuse)
    ##########################################
    #lo=tf.argmax(logits,1)
    #L =tf.argmax(t_labels,1)

    correct_prediction = tf.equal(tf.argmax(logits,1),tf.argmax(labels,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
    ################################################

    
    return  loss,cross_entropy_mean,accuracy
    
def train (FLAGS,total_loss,global_step):

    num_batches_per_epoch = NUM_EXAMPLES_PER_EPOXH_FOR_TRAIN/FLAGS.batch_size
    decay_steps = int(num_batches_per_epoch*NUM_EPOCH_PER_DECAY)

    #decay the learning rate exponentially based on the number of steps
    #隨著迭代過程衰減學習率
    lr=tf.train.exponential_decay(INITIAL_LEARNING_RATE,global_step,decay_steps,LEARNING_RATE_DECAY_FACTOR,staircase=True)
    tf.summary.scalar('learning_rate',lr)

    #滑動平均 of all [losses] and associated summaries
    loss_averages_op=_add_loss_summaries(total_loss)

    #計算梯度
    with tf.control_dependencies([loss_averages_op]):
        opt=tf.train.GradientDescentOptimizer(lr)
        grads=opt.compute_gradients(total_loss)

    #apply gradients
    apply_gradient_op=opt.apply_gradients(grads,global_step=global_step)
    #This is the second part of `minimize()`. It returns an `Operation` that applies gradients.

    #add histogram
    for grad,var in grads:
        if grad is not None:
            tf.summary.histogram(var.op.name+'/gradients',grad)

    # Track the moving averages of all trainable variables.
    variable_averages = tf.train.ExponentialMovingAverage(
        MOVING_AVERAGE_DECAY, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())

    with tf.control_dependencies([apply_gradient_op, variables_averages_op]):#why use it? I think all antecedent ops are not only just activated first 
                                                                            #,but also could be representation that u wanna ops to be congreated   
        train_op = tf.no_op(name='train')

    return train_op

def _add_loss_summaries(total_loss):
    """
    Add summaries for losses in CIFAR-10 model
    Generates moving average for all losses and associated summaries of visualizing the performnce of the network
    :param total_loss:Total loss from loss()
    :return:
    loss_averages_op: op for generating moving averages of losses
    """
    #計算moving average of all individual losses and the total loss
    #MovingAverage為滑動平均，計算方法：對於一個給定的數列，首先設定一個固定的值k，然後分別計算第1項到第k項，第2項到第k+1項，第3項到第k+2項的平均值，依次類推。
    loss_averages=tf.train.ExponentialMovingAverage(0.9,name='avg')
    losses=tf.get_collection('losses')
    

    loss_averages_op=loss_averages.apply(losses+[total_loss])
    """ with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        print(sess.run(losses))
        print(sess.run(total_loss))
        print(loss_averages_op) """

    #給每一個單獨的losses和total loss attach a scalar summary;do the same
    #for the averaged version of the losses
    for l in losses+[total_loss]:

        tf.summary.scalar(l.op.name+'(raw)',l)
        tf.summary.scalar(l.op.name,loss_averages.average(l))

    return loss_averages_op