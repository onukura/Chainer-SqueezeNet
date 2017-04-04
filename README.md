# SqueezeNet chainer version

1.Download SqueezeNet ver1.1 caffemodel from https://github.com/DeepScale/SqueezeNet/tree/master/SqueezeNet_v1.1    
2.model = SqueezeNet(output)    
3.func = CaffeFunction(caffemodel_path)    
4.convert_squeezenet_v11(src=func, dst=model)    
