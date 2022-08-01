# Implementation of 'Pruning Filters for Efficient ConvNets' 

### Creator: Derek Montanez - Texas State University - Computer Science - dnm89@txstate.edu 

### Summer 2022 DIMACS Rutgers REU 

<p> This work is an implementation of 'Pruning Filters for Efficient ConvNets' a structured pruning method for compression of convolutional neural networks. This work prunes a user set percentage (X) of filters in each convolutional layer in the model 'SCNNB'. A filter is to be pruned dependent on the l1 norm of the filter. Where the lowest (X) percentage of l1 norm filters in each convolutional layer are pruned. This process induces a domino effect on the next convolutional layer. If a filter is to be pruned from convoltuional layer i, then the corresponding feature map in convolutional layer i+1 is pruned as well. Therefore the i+1 convolutional layers corresponding filter channels are deleted as well. The paper, introduces two methods different procedures to go about pruning. A one shot pruning method, and a iterative prunign method. The former, sets a pre defined pruning percentage for each convolutional layer, and and prunes the entire model in a forward pass through the model, and retraining is done after the pruning procedure. The latter, prunes a percentage of filters in a layer, and a retraining process begins for the model, and this process is performed on all convolutional layers of the model. This implementation, is carried out using the former method (one-shot). The paper introduces two types of pruning approaches for the one shot approach, an independent and greedy. The former, prunes filters in a layer, wihtout depending on the filters pruned in the previous layer. While the latter, prunes filters in a layer, with consideration of filters that have been pruned in previous layers. We will use the independent prunign strategy in this approach. </p>



### 'Pruning Filters for Efficient ConvNets' 
<p> To learn more about the pruning algorithm ----> //link here to paper // </p>

<h3> Capabilities and Limitations </h3> 
<p> This program is currently able to prune the model SCNNB, a shallow convolutional neural network, using a set precision of filter percentage for every layer of the model. The datasets used in this program include CIFAR10 and MNIST Fashion, two image classification datasets. I would like to extend this pruning method to be applicable to more datasets, and especially being able to be implemented using other neural network models such as VGG16 or AlexNet. </p>


### Installations needed
				
> Python 3.7 - 3.10

> Ubuntu 16.04 

> Windows 7 or Later	

> Installation of tensorflow 

> Installation of keras

### Command line arguments needed

> dataset = 'fashion' or 'cifar10'

> lr = float in range x, x > 0

> batchSz = int in range x, x > 0

> Epoch1 = int in range x, x > 0 

> Epoch2 = int in range x, x > 0

> filterPercentage = float in range x, x > 0 and x < 1 																		
### Syntax
### If you would like to run the process after logging off, the 'nohup' command is needed, as well as the ' > outputFile.txt' where output.txt is any predfined txt file
> nohup python3 Formal.py dataset lr batchSz Epoch1 Epoch2 fitlerPercentage > outputFile.txt 

### Write to the command line
> python3 Formal.py dataset lr batchSz Epoch1 Epoch2 filterPercentage 

### Example
> nohup: nohup python3 Formal.py fashion 0.001 128 35 30 0.4 > outputFile.txt 

> w/out nohup: python3 Formal.py fashion 0.001 128 35 30 0.4



