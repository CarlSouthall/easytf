# Usage

This file contains information regarding the usage of easytf.

## easytf.RNN (Recurrent neural networks)

Included in this class:
- single directional and bi-directional configurations.
- tanh, LSTM, GRU, and LSTMP cell architectures.
- soft attention mechanisms.

### Initialization

```Python
network = RNN(training_data, training_labels, validation_data, validation_labels, mini_batch_locations, network_save_filename, minimum_epoch = 5, maximum_epoch = 10, n_hidden = [100,100], n_classes = 2, cell_type = 'LSTMP', configuration = ''B', attention_number = 0, dropout = 0.25, init_method = 'zero', truncated = 1000, optimizer ='Adam', learning_rate = 0.003, display_train_loss ='True', display_accuracy='True')
```

`training_data`  : training data features used to train the network [dim1 x dim2] 

`training_labels` : 	training labels corresponding to training_data [dim1 x dim2]

`validation_data` : 	validation data features used to validate the network [dim1 x dim2]

`validation_labels` : 	validation labels corresponding to validation_data [dim1 x dim2]

`mini_batch_locations` : the locations in training_data for each of the observations in each batch. Second dimensions defines the batch size.

`network_save_filename` :	 the filename named used to save the network configuration and parameters.

`minimum_epoch` :	the minimum number of training epochs.

`maximum_epoch` : 	the maximum number of training epochs.

`n_hidden`: 		the number of layers in each hidden layer.

`n_layers` :		the number of layers (could remove this)

`n_classes` :	the number of output classes / neurons in the output layer.

`cell_type` :		the cell architectures used for the hidden layers

`configuration` :		the configuration used

`attention_number` :     the number of soft attention connections 

`dropout` :		the dropout probability.

`init_method` :	the initialization technique used.

`truncated` : 	the truncation number used in truncated back_propagation

`optimizer` :		the optimizer used

`learning_rate` : 	the learning rate used

`display_train_loss` : 	to whether or not to display the train 	

`display_accuracy` :	whether or not to display the accuracies

### Functions

```
network.create()
```
Creates the network using the given parameters.

```
network.train()
```
Trains the network using the given training and validation data.

```
out=network.implement(Test_data)
```

Runs the test data through the network.


### Examples

```python
import easytf

network=easytf.RNN(train_data, train_targ, val_data, val_targ, filename)
network.train()
out=network.implement(test_data)
```

Creates, trains and implements a bidirectional recurrent neural network with LSTMP cell architectures.

```python
import easytf

network=easytf.RNN(train_data, train_targ, val_data, val_targ, filename, cells='tanh', configuration='R')
network.train()
out=network.implement(test_data)
```

Creates, trains and implements a recurrent neural network with tanh cell architectures.

```python
import easytf

network=easytf.RNN(train_data, train_targ, val_data, val_targ, filename, cells='GRU', attention_number=2)
network.train()
out=network.implement(test_data)
```

Creates, trains and implements a bidirectional recurrent neural network with GRU cell architectures containing soft attention connections with an attention number of 2.





## easytf.CNN (Convolutional neural networks)

Included in this class:
- 2d convolution

- 3d convolution

### Initialization


```Python
network = CNN(training_data, training_labels, validation_data, validation_labels, mini_batch_locations, network_save_filename, minimum_epoch = 5, maximum_epoch = 100, learning_rate = 0.003, n_classes = 2, optimizer = 'Adam', conv_filter_shapes = [[5,5,1,5],[5,5,5,10]], conv_strides = [[1,1,1,1],[1,1,1,1]], pool_window_sizes=[1,1,2,1],[1,1,2,1]], fc_layer_size = [100], dropout = 0.25, pad = 'SAME', display_accuracy='True', display_train_loss='True', frames_either_side = [[2,2],[0,0]], input_stride_size = [1, 1024])
```

`training_data`  : training data features used to train the network [dim1 x dim2] 

`training_labels` : 	training labels corresponding to training_data [dim1 x dim2]

`validation_data` : 	validation data features used to prevent over fitting [dim1 x dim2]

`validation_labels` : 	validation labels corresponding to validation_data [dim1 x dim2]

`mini_batch_locations` :	the locations of the training_data used for each observation.

`network_save_filename` :	 the filename named used to save the network configuration and parameters.

`minimum_epoch` :	the minimum number of training epochs.

`maximum_epoch` : 	the maximum number of training epochs.

`learning_rate` : 	the learning rate used.

`n_classes` :	the number of output classes / neurons in the output layer.

`optimizer` :		the optimizer used

`conv_filter_shapes` 	: 	the filter shape sizes used in each convolutional layer.

`conv_strides`	: 	the stride lengths used in each convolutional layer.

`pool_window_sizes` :	the size of the windows used in the max pooling layers.

`fc_layer_size`	: 	number of neurons used in each of the fully-connected layers.

`dropout` :		the dropout probability.

`init_method` :	the initialization technique used.

`display_train_loss` : 	to whether or not to display the train.

`display_accuracy` :	whether or not to display the accuracies.

`frames_either_side`	:	the frames either side included in the input features.

`input_stride_size`	:	the stride length used in the input features, determines whether 2d or 3d convolution is implement.


### Functions

```
network.create()
```
Creates the network using the given parameters.

```
network.train()
```
Trains the network using the given training and validation data.

```
out=network.implement(Test_data)
```

Runs the test data through the network.


### Examples

```python
import easytf

network=easytf.CNN(train_data, train_targ, val_data, val_targ, mini_batch_locations, filename)
network.train()
out=network.implement(test_data)
```

Creates, trains and implements a two layered 2d convolutional neural network.

```python
import easytf

network=easytf.CNN(train_data, train_targ, val_data, val_targ, mini_batch_locations, filename, conv_filter_shapes = [[5,5,5,1,5],[5,5,5,5,10]], conv_strides = [[1,1,1,1,1],[1,1,1,1,1]], pool_window_sizes=[1,1,2,1,1],[1,1,2,1,1]] frames_either_side = [[2,2],[0,0],[0,0]], input_stride_size = [1, 1024,1])
network.train()
out=network.implement(test_data)
```

Creates, trains and implements a two layered 3d convolutional neural network.






