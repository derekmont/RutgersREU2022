# Implementation of 'Pruning Filters for Efficient ConvNets' 

### Creator: Derek Montanez - Texas State University - Computer Science - dnm89@txstate.edu 

### Summer 2022 DIMACS Rutgers REU - Exploring the tradeoffs between compression and accuracy.

### Motivation
<p> With the surgence of neural networks, and the impressive tasks they acccomplish in our everyday life, we encounter a problem. The problem is the size of state of the art neural networks, impose large compuational and memory costs. It is difficult to deploy and use deep neural networks on edge devices, because of the computational and memory constraints that the edge devices have. We would like to compress deep neural networks into smaller models that work as well or better than the original deep model. The smaller model will be easier to deploy and use on edge devices, while hopefully not damaging the performance of the model too drastically. Deep neural network compression methods have been proposed since the late 1980's. When applying these compression methods, it is impportant to analyze and justify the compression vs. accuracy tradeoff, and this pruning implementation will help in analyzing this problem. 
	
### Testings 
### MNIST Fashion Dataset - Number of Parameters after Pruning at Different Levels vs. Accuracy
![fashion](https://user-images.githubusercontent.com/98001990/182316067-e8de044e-6052-4249-b6d2-36ae827aa925.png)

### CIFAR10 Dataset - Number of Parameters after Pruning at Different Levels vs. Accuracy
![cifar](https://user-images.githubusercontent.com/98001990/182315962-93fb0cf3-eeb5-40a8-8b2e-c8e854e91b86.png)

### Implementation
<p> This work is an implementation of 'Pruning Filters for Efficient ConvNets' a structured pruning method for compression of convolutional neural networks. This work prunes a user set percentage (X) of filters in each convolutional layer in the model 'SCNNB'. A filter is to be pruned dependent on the l1 norm of the filter. Where the lowest (X) percentage of l1 norm filters in each convolutional layer are pruned. This process induces a domino effect on the next convolutional layer. If a filter is to be pruned from convoltuional layer i, then the corresponding feature map in convolutional layer i+1 is pruned as well. Therefore the i+1 convolutional layers corresponding filter channels are deleted as well. The paper, introduces two methods different procedures to go about pruning. A one shot pruning method, and a iterative pruning method. The former, sets a pre defined pruning percentage for each convolutional layer, and prunes the entire model in a forward pass through the model, and retraining is done after the pruning procedure. The latter, prunes a percentage of filters in a layer, and a retraining process begins for the model, and this process is performed on all convolutional layers of the model. This implementation, is carried out using the former method (one-shot). The paper also introduces two types of pruning approaches for the one shot approach, an independent and greedy implementation. The former, prunes filters in a layer, wihtout depending on the filters pruned in the previous layer. While the latter, prunes filters in a layer, with consideration of filters that have been pruned in previous layers. We will use the independent pruning strategy in this implementation. 'Pruning Filters for Efficient ConvNets' follows a train, prune, and retrain process to makeup for accuracy loss after pruning. They begin training for X epochs, and retraining is done for X' epochs, where X' < X </p>


### 'Pruning Filters for Efficient ConvNets' and SCNNB
> To learn more about the pruning algorithm click here!! ----> https://arxiv.org/abs/1608.08710 
> To learn more about he model used in this implementation!! ----> https://link.springer.com/article/10.1007/s42452-019-1903-4

<h3> Capabilities and Limitations </h3> 
<p> This program is currently able to prune the model SCNNB, a shallow convolutional neural network, using a set precision of filter percentage for every layer of the  model. The datasets used in this program include CIFAR10 and MNIST Fashion, two image classification datasets. I would like to extend this pruning method to be  applicable to more datasets, and especially being able to be implemented using other neural network models such as VGG16 or AlexNet. </p>


### Installations needed
				
> Python 3.7 - 3.10

> Ubuntu 16.04 or later

> Windows 7 or Later	

> Installation of tensorflow ---> https://www.tensorflow.org/install/pip

> Installation of keras ---> https://www.tutorialspoint.com/keras/keras_installation.htm

> Installation of numpy

> Installation of pandas

### Command line arguments needed

> dataset = 'fashion' or 'cifar10'

> lr = float in range x, x > 0

> batchSz = int in range x, x > 0

> Epoch1 = int in range x, x > 0 

> Epoch2 = int in range x, x > 0

> filterPercentage = float in range x, 0 <= x < 1 																		
### Syntax
### Baseline model - No Pruning 
> nohup python3 Formal.py dataset lr batchSz Epoch1 Epoch2 0.0 > outputFile.txt

### Pruned Version
> nohup python3 Formal.py dataset lr batchSz Epoch1 Epoch2 fitlerPercentage > outputFile.txt 

### Example
> nohup: nohup python3 Formal.py fashion 0.001 128 35 30 0.4 > outputFile.txt 
