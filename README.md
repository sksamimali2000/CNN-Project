# Convolutional Neural Network (CNN) for MNIST Digit Classification

## Overview
This project demonstrates a Convolutional Neural Network (CNN) implementation in TensorFlow to classify handwritten digits from the MNIST dataset.

## Dataset
- **MNIST**: 28x28 grayscale images of handwritten digits (0–9).
- Training, test, and validation sets provided by `tensorflow.examples.tutorials.mnist`.

## Model Architecture

### Input
- Input Dimensions: 28x28 pixels (flattened to 784).
- Channels: 1 (grayscale).

### Convolutional Layers
1. **Conv Layer 1**
   - Filters: 32
   - Kernel Size: 5x5
   - Stride: 1
   - Activation: ReLU
   - Max Pooling: 2x2

2. **Conv Layer 2**
   - Filters: 64
   - Kernel Size: 5x5
   - Stride: 1
   - Activation: ReLU
   - Max Pooling: 2x2

### Fully Connected Layer
- Hidden Neurons: 1024
- Activation: ReLU
- Dropout applied with `keep_prob`.

### Output Layer
- 10 neurons (for digit classes 0–9)
- No activation applied directly, used for softmax computation during loss calculation.

## Training Setup
- Loss Function: Softmax Cross Entropy with logits
- Optimizer: Adam with learning rate 0.01
- Batch Size: 100
- Epochs: 25
- Dropout keep probability during training: 0.8

## Training Process
- Dataset divided into batches.
- Total cost calculated per epoch and printed.
  
## Evaluation
- After training, predictions computed by taking the `argmax` of logits.
- Number of correct predictions on the test dataset calculated.

## Example Output
```plaintext
Total cost per epoch printed during training.
Final correct predictions count on test set: <correct_preds.sum()>
```

Usage
Data Loading
```Python
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
```

Define Model

Convolution + MaxPooling operations defined in helper functions.

Model built using tf.nn.conv2d, tf.nn.max_pool, and dense layers.

Train Model
```Python
for i in range(25):
    for j in range(num_batches):
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        c, _ = sess.run([cost, optimize], feed_dict={x: batch_x, y: batch_y, keep_prob: 0.8})

Evaluate Model
predictions = tf.argmax(pred, 1)
correct_labels = tf.argmax(y, 1)
correct_predictions = tf.equal(predictions, correct_labels)
correct_preds_sum = sess.run(correct_predictions.sum(), feed_dict={
    x: mnist.test.images,
    y: mnist.test.labels,
    keep_prob: 1.0
})
```

Notes

This implementation uses TensorFlow 1.x style (with sessions and placeholders).

Suitable for learning basic CNN structure in TensorFlow.
