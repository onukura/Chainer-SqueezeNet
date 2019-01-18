# SqueezeNet 
[Chainer](https://github.com/pfnet/chainer) implimentation of  [SqueezeNet](http://arxiv.org/abs/1602.07360).  
This repository contains v1.0 and v1.1.  
Official trained caffemodel will be downloaded from [DeepScale/SqueezeNet](https://github.com/DeepScale/SqueezeNet).  

## Requirement
	Chainer == 5.1.0

## Usage

	import numpy as np
	import chainer

	from squeezenet import SqueezeNet_V10
	from squeezenet import SqueezeNet_V11
	
	sq10 = SqueezeNet_V10()
	sq11 = SqueezeNet_V11()
	
	x = chainer.Variable(np.ones((1, 3, 224, 224), np.float32))
	sq10(x, layer=['prob'])
	sq10(x, layer=['prob'])
	
