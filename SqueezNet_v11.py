import chainer
import chainer.functions as F
import chainer.links as L

class SqueezeNet_v11(chainer.Chain):

    insize = 227

    def __init__(self, output):
        super(SqueezeNet_v11, self).__init__(
            #conv1=L.Convolution2D(3, 96, 7, stride=2),
            conv1=L.Convolution2D(3, 64, ksize=3, stride=2),
            
            #fire2=Fire(96, 16, 64, 64),
            fire2_squeeze_11=L.Convolution2D(64, 16, 1),
            fire2_expand_11=L.Convolution2D(16, 64, 1),
            fire2_expand_33=L.Convolution2D(16, 64, 3, pad=1),
            fire2_bn=L.BatchNormalization(128),
            			
            #fire3=Fire(128, 16, 64, 64),
            fire3_squeeze_11=L.Convolution2D(128, 16, 1),
            fire3_expand_11=L.Convolution2D(16, 64, 1),
            fire3_expand_33=L.Convolution2D(16, 64, 3, pad=1),
            fire3_bn=L.BatchNormalization(128),
            			
            #fire4=Fire(128, 16, 128, 128),
            fire4_squeeze_11=L.Convolution2D(128, 32, 1),
            fire4_expand_11=L.Convolution2D(32, 128, 1),
            fire4_expand_33=L.Convolution2D(32, 128, 3, pad=1),
            fire4_bn=L.BatchNormalization(256),		
            			
            #fire5=Fire(256, 32, 128, 128),
            fire5_squeeze_11=L.Convolution2D(256, 32, 1),
            fire5_expand_11=L.Convolution2D(32, 128, 1),
            fire5_expand_33=L.Convolution2D(32, 128, 3, pad=1),
            fire5_bn=L.BatchNormalization(256),
            			
            #fire6=Fire(256, 48, 192, 192),
            fire6_squeeze_11=L.Convolution2D(256, 48, 1),
            fire6_expand_11=L.Convolution2D(48, 192, 1),
            fire6_expand_33=L.Convolution2D(48, 192, 3, pad=1),
            fire6_bn=L.BatchNormalization(384),
            			
            #fire7=Fire(384, 48, 192, 192),
            fire7_squeeze_11=L.Convolution2D(384, 48, 1),
            fire7_expand_11=L.Convolution2D(48, 192, 1),
            fire7_expand_33=L.Convolution2D(48, 192, 3, pad=1),
            fire7_bn=L.BatchNormalization(384),
            			
            #fire8=Fire(384, 64, 256, 256),
            fire8_squeeze_11=L.Convolution2D(384, 64, 1),
            fire8_expand_11=L.Convolution2D(64, 256, 1),
            fire8_expand_33=L.Convolution2D(64, 256, 3, pad=1),
            fire8_bn=L.BatchNormalization(512),
            			
            #fire9=Fire(512, 64, 256, 256),
            fire9_squeeze_11=L.Convolution2D(512, 64, 1),
            fire9_expand_11=L.Convolution2D(64, 256, 1),
            fire9_expand_33=L.Convolution2D(64, 256, 3, pad=1),
            fire9_bn=L.BatchNormalization(512),
            			
            conv10=L.Convolution2D(512, 1000, 1),
            
            #additional part
            fc11 = L.Linear(1000, output)
        )
        self.train = True
		
    def forward(self, x, train):
        #conv1
        h = F.relu(self.conv1(x))
		
        #pooling1
        h = F.max_pooling_2d(h, 3, stride=2)
		
        #fire2
        h = F.relu(self.fire2_squeeze_11(h))
        h_1 = self.fire2_expand_11(h)
        h_3 = self.fire2_expand_33(h)
        h_out = F.concat((h_1, h_3), axis=1)
        h = F.relu(self.fire2_bn(h_out))
		
        #fire3
        h = F.relu(self.fire3_squeeze_11(h))
        h_1 = self.fire3_expand_11(h)
        h_3 = self.fire3_expand_33(h)
        h_out = F.concat((h_1, h_3), axis=1)
        h = F.relu(self.fire3_bn(h_out))
		
        #pooling2
        h = F.max_pooling_2d(h, 3, stride=2)
		
        #fire4
        h = F.relu(self.fire4_squeeze_11(h))
        h_1 = self.fire4_expand_11(h)
        h_3 = self.fire4_expand_33(h)
        h_out = F.concat([h_1, h_3], axis=1)
        h = F.relu(self.fire4_bn(h_out))
		
        #fire5
        h = F.relu(self.fire5_squeeze_11(h))
        h_1 = self.fire5_expand_11(h)
        h_3 = self.fire5_expand_33(h)
        h_out = F.concat((h_1, h_3), axis=1)
        h = F.relu(self.fire5_bn(h_out))
		
        #pooling3
        h = F.max_pooling_2d(h, 3, stride=2)
		
        #fire6
        h = F.relu(self.fire6_squeeze_11(h))
        h_1 = self.fire6_expand_11(h)
        h_3 = self.fire6_expand_33(h)
        h_out = F.concat((h_1, h_3), axis=1)
        h = F.relu(self.fire6_bn(h_out))
		
        #fire7
        h = F.relu(self.fire7_squeeze_11(h))
        h_1 = self.fire7_expand_11(h)
        h_3 = self.fire7_expand_33(h)
        h_out = F.concat((h_1, h_3), axis=1)
        h = F.relu(self.fire7_bn(h_out))
		
        #fire8
        h = F.relu(self.fire8_squeeze_11(h))
        h_1 = self.fire8_expand_11(h)
        h_3 = self.fire8_expand_33(h)
        h_out = F.concat((h_1, h_3), axis=1)
        h = F.relu(self.fire8_bn(h_out))

        #fire9
        h = F.relu(self.fire9_squeeze_11(h))
        h_1 = self.fire9_expand_11(h)
        h_3 = self.fire9_expand_33(h)
        h_out = F.concat((h_1, h_3), axis=1)
        h = F.relu(self.fire9_bn(h_out))

        #dropout1
        h = F.dropout(h, ratio=0.5, train=train)
        
        #conv10
        h = F.relu(self.conv10(h))
		
        #pooling4
        h = F.average_pooling_2d(h, 13)
        
        h = F.reshape(h, (-1, 1000))
        
        #additional part
        h = F.dropout(h, 0.5, train=train)
        y = self.fc11(h)
		
        return y
        
    def __call__(self, x, t, train=True):
        y = self.forward(x, train=train)
        self.loss = F.softmax_cross_entropy(y, t)
        self.accuracy = F.accuracy(y, t)        
        return self.loss
#########################
### SqueezeNet ###########
#########################
