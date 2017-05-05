# Usage

This file contains information regarding the usage of easytf.

## easytf.RNN (Recurrent neural networks)

Included in this class:
- single directional and bi-directional configurations.
- tanh, LSTM, GRU, and LSTMP cell architectures.
- soft attention mechanisms.

### Initialization

```Python
network = RNN(training_data, training_labels, validation_data, validation_labels, network_save_filename, minimum_epoch = 5, maximum_epoch = 10, n_hidden = [100,100], n_classes = 2, cell_type = 'LSTMP', configuration = ''B', attenion_number = 2, init_method = 'zero', truncated = 1000, optimizer ='Adam', learning_rate = 0.003, display_train_loss ='True', display_accuracy='True')
```

`training_data`  : training data features used to train the network [dim1 x dim2] 

`training_labels` : 	training labels corresponding to training_data [dim1 x dim2]

`validation_data` : 	validation data features used to validate the network [dim1 x dim2]

`validation_labels` : 	validation labels corresponding to validation_data [dim1 x dim2]

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



## easytf.CNN (Convolutional neural networks)

Included in this class:
- 2d convolution

- 3d convolution

### Initialization


```Python
network = CNN(training_data, training_labels, validation_data, validation_labels, mini_batch_locations, network_save_filename, minimum_epoch = 5, maximum_epoch = 100, learning_rate = 0.003, n_classes = 2, optimizer = 'Adam', conv_filter_shapes = [[5,5,1,5],[5,5,5,10]], conv_strides = [[1,1,1,1],[1,1,1,1]], pool_window_sizes=[1,1,2,1],[1,1,2,1]], fc_layer_size = [100], dropout = 0.25, pad = 'SAME', display_accuracy='True', display_train_loss='True', frames_either_side = [[2,2],[0,0]], input_stride_size = [1, 1025], dim = '2d')
```

`training_data`  : training data features used to train the network [dim1 x dim2] 

`training_labels` : 	training labels corresponding to training_data [dim1 x dim2]

`validation_data` : 	validation data features used to validate the network [dim1 x dim2]

`validation_labels` : 	validation labels corresponding to validation_data [dim1 x dim2]

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

```Python
from ADTLib.models import ADTBDRNN

Filenames='Drumfile.wav'
X=ADTBDRNN(Filenames)
```
Output stored in variable X ordered by time.
  
```Python
from ADTLib.models import ADTBDRNN

Filenames=['Drumfile.wav','Drumfile1.wav']
ADTBDRNN(Filenames,out_sort='instrument',ret='no',out_text='yes',savedir='Desktop')
```
Output ordered by instrument printed to a text file on the desktop.



