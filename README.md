# CNN-by-tensorflow
Extend more tools you would need in the Advanced Convolutional Neural Networks from official Tensorflow
----------------------------------------------------------------------------------------------------------

The inference Model is from https://github.com/tensorflow/models/blob/master/tutorials/image/cifar10/cifar10.py

Objective: 
1.transfer image to be TFrecord from. 
2.add tf.dataset method instead of the feed_dict.
3.Add the computing validation while training data.

Data: 
I use the company provided data so that I can't share it with you . 
But you will be able to utlized Creat_tfrecords.py to transfer it to TFrecord files for your image data.

training model: 
supports the RGB image training. need to revise the depths of numbers of classes in train.py/loss that how many classes in your case .
