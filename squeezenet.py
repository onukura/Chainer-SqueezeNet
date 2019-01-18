# -*- coding: utf-8 -*-

import collections
import os
import sys

import chainer
import chainer.functions as F
import chainer.links as L
from chainer.dataset import download
from chainer.serializers import npz

class FireBlock(chainer.Chain):

    def __init__(self, in_ch, mid_ch, out_ch):
        super(FireBlock, self).__init__()
        with self.init_scope():
            self.squeeze_11 = L.Convolution2D(in_ch,  mid_ch, 1)
            self.expand_11  = L.Convolution2D(mid_ch, out_ch, 1)
            self.expand_33  = L.Convolution2D(mid_ch, out_ch, 3, pad=1)
            self.bn         = L.BatchNormalization(out_ch * 2)

    def __call__(self, x):
        h = F.relu(self.squeeze_11(x))
        h = F.concat((self.expand_11(h),
                      self.expand_33(h)), axis=1)
        return F.relu(self.bn(h))


class SqueezeNet(chainer.Chain):

    insize = 227

    def __init__(self, c1_ch, c1_k, pretrained_model=None):
        super(SqueezeNet, self).__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(3, c1_ch, ksize=c1_k, stride=2)
            self.fire2 = FireBlock(c1_ch,  16,  64)
            self.fire3 = FireBlock(128, 16,  64)
            self.fire4 = FireBlock(128, 32, 128)
            self.fire5 = FireBlock(256, 32, 128)
            self.fire6 = FireBlock(256, 48, 192)
            self.fire7 = FireBlock(384, 48, 192)
            self.fire8 = FireBlock(384, 64, 256)
            self.fire9 = FireBlock(512, 64, 256)
            self.conv10 = L.Convolution2D(512, 1000, 1)

        if pretrained_model == 'v10':
            _retrieve(
                'squeezenet_v10.npz',
                'https://github.com/DeepScale/SqueezeNet/SqueezeNet_v1.0/squeezenet_v1.0.caffemodel',
                self)
        if pretrained_model == 'v11':
            _retrieve(
                'squeezenet_v11.npz',
                'https://github.com/DeepScale/SqueezeNet/SqueezeNet_v1.1/squeezenet_v1.1.caffemodel',
                self)
        elif pretrained_model:
            npz.load_npz(pretrained_model, self)

    @property
    def available_layers(self):
        return list(self.functions.keys())
    
    @classmethod
    def convert_caffemodel_to_npz(cls, path_caffemodel, path_npz):
        """Converts a pre-trained caffemodel to a chainer model.

        Args:
            path_caffemodel (str): Path of the pre-trained caffemodel.
            path_npz (str): Path of the converted chainer model.
        """

        # As CaffeFunction uses shortcut symbols,
        # we import CaffeFunction here.
        from chainer.links.caffe.caffe_function import CaffeFunction
        caffemodel = CaffeFunction(path_caffemodel)
        chainermodel = cls(pretrained_model=None)
        _transfer_squeezenet(caffemodel, chainermodel)
        npz.save_npz(path_npz, chainermodel, compression=False)

    def forward(self, x, layers=['prob']):
        """Computes all the feature maps specified by ``layers``.

        Args:
            x (~chainer.Variable): Input variable.
            layers (list of str): The list of layer names you want to extract.
            test (bool): If ``True``, BarchNormalization runs in test mode.

        Returns:
            Dictionary of ~chainer.Variable: A directory in which
            the key contains the layer name and the value contains
            the corresponding feature map variable.

        """

        h = x
        activations = {}
        target_layers = set(layers)
        for key, funcs in self.functions.items():
            if len(target_layers) == 0:
                break
            for func in funcs:
                h = func(h)
            if key in target_layers:
                activations[key] = h
                target_layers.remove(key)
        return activations

    def __call__(self, x, layers=['prob']):
        return self.forward(x, layers)


class SqueezeNet_V10(SqueezeNet):

    def __init__(self, pretrained_model='auto'):
        if pretrained_model == 'auto': 
            pretrained_model = 'v10'
        super(SqueezeNet_V10, self).__init__(96, 7, pretrained_model)

    @property
    def functions(self):
        return collections.OrderedDict([
            ('conv1', [self.conv1, F.relu]),
            ('pool1', [lambda x: F.max_pooling_2d(x, 3, stride=2)]),
            ('fire2', [self.fire2]),
            ('fire3', [self.fire3]),
            ('fire4', [self.fire4]),
            ('pool2', [lambda x: F.max_pooling_2d(x, 3, stride=2)]),
            ('fire5', [self.fire5]),
            ('fire6', [self.fire6]),
            ('fire7', [self.fire7]),
            ('fire8', [self.fire8]),
            ('pool3', [lambda x: F.max_pooling_2d(x, 3, stride=2)]),
            ('fire9', [self.fire9, F.dropout]),
            ('conv10', [self.conv10, F.relu]),
            ('pool4', [lambda x: F.average_pooling_2d(x, 13)]),
            ('prob',  [lambda x: F.reshape(x, (-1, 1000))]),
        ])


class SqueezeNet_V11(SqueezeNet):

    def __init__(self, pretrained_model='auto'):
        if pretrained_model == 'auto':
            pretrained_model = 'v11'
        super(SqueezeNet_V11, self).__init__(64, 3, pretrained_model)

    @property
    def functions(self):
        return collections.OrderedDict([
            ('conv1', [self.conv1, F.relu]),
            ('pool1', [lambda x: F.max_pooling_2d(x, 3, stride=2)]),
            ('fire2', [self.fire2]),
            ('fire3', [self.fire3]),
            ('pool2', [lambda x: F.max_pooling_2d(x, 3, stride=2)]),
            ('fire4', [self.fire4]),
            ('fire5', [self.fire5]),
            ('pool3', [lambda x: F.max_pooling_2d(x, 3, stride=2)]),
            ('fire6', [self.fire6]),
            ('fire7', [self.fire7]),
            ('fire8', [self.fire8]),
            ('fire9', [self.fire9, F.dropout]),
            ('conv10', [self.conv10, F.relu]),
            ('pool4', [lambda x: F.average_pooling_2d(x, 13)]),
            ('prob',  [lambda x: F.reshape(x, (-1, 1000))]),
        ])


def _transfer_squeezenet(src, dst):
    # copy parameters from caffemodel into chainer model
    print('start copy params.')
    caffe_model = src

    dst.conv1.W.data = caffe_model['conv1'].W.data
    dst.conv1.b.data = caffe_model['conv1'].b.data
    print('copy conv1')

    dst.fire2.squeeze_11.W.data = caffe_model['fire2/squeeze1x1'].W.data
    dst.fire2.squeeze_11.b.data = caffe_model['fire2/squeeze1x1'].b.data
    dst.fire2.expand_11.W.data = caffe_model['fire2/expand1x1'].W.data
    dst.fire2.expand_11.b.data = caffe_model['fire2/expand1x1'].b.data
    dst.fire2.expand_33.W.data = caffe_model['fire2/expand3x3'].W.data
    dst.fire2.expand_33.b.data = caffe_model['fire2/expand3x3'].b.data
    print('copy fire2')

    dst.fire3.squeeze_11.W.data = caffe_model['fire3/squeeze1x1'].W.data
    dst.fire3.squeeze_11.b.data = caffe_model['fire3/squeeze1x1'].b.data
    dst.fire3.expand_11.W.data = caffe_model['fire3/expand1x1'].W.data
    dst.fire3.expand_11.b.data = caffe_model['fire3/expand1x1'].b.data
    dst.fire3.expand_33.W.data = caffe_model['fire3/expand3x3'].W.data
    dst.fire3.expand_33.b.data = caffe_model['fire3/expand3x3'].b.data
    print('copy fire3')

    dst.fire4.squeeze_11.W.data = caffe_model['fire4/squeeze1x1'].W.data
    dst.fire4.squeeze_11.b.data = caffe_model['fire4/squeeze1x1'].b.data
    dst.fire4.expand_11.W.data = caffe_model['fire4/expand1x1'].W.data
    dst.fire4.expand_11.b.data = caffe_model['fire4/expand1x1'].b.data
    dst.fire4.expand_33.W.data = caffe_model['fire4/expand3x3'].W.data
    dst.fire4.expand_33.b.data = caffe_model['fire4/expand3x3'].b.data
    print('copy fire4')

    dst.fire5.squeeze_11.W.data = caffe_model['fire5/squeeze1x1'].W.data
    dst.fire5.squeeze_11.b.data = caffe_model['fire5/squeeze1x1'].b.data
    dst.fire5.expand_11.W.data = caffe_model['fire5/expand1x1'].W.data
    dst.fire5.expand_11.b.data = caffe_model['fire5/expand1x1'].b.data
    dst.fire5.expand_33.W.data = caffe_model['fire5/expand3x3'].W.data
    dst.fire5.expand_33.b.data = caffe_model['fire5/expand3x3'].b.data
    print('copy fire5')

    dst.fire6.squeeze_11.W.data = caffe_model['fire6/squeeze1x1'].W.data
    dst.fire6.squeeze_11.b.data = caffe_model['fire6/squeeze1x1'].b.data
    dst.fire6.expand_11.W.data = caffe_model['fire6/expand1x1'].W.data
    dst.fire6.expand_11.b.data = caffe_model['fire6/expand1x1'].b.data
    dst.fire6.expand_33.W.data = caffe_model['fire6/expand3x3'].W.data
    dst.fire6.expand_33.b.data = caffe_model['fire6/expand3x3'].b.data
    print('copy fire6')

    dst.fire7.squeeze_11.W.data = caffe_model['fire7/squeeze1x1'].W.data
    dst.fire7.squeeze_11.b.data = caffe_model['fire7/squeeze1x1'].b.data
    dst.fire7.expand_11.W.data = caffe_model['fire7/expand1x1'].W.data
    dst.fire7.expand_11.b.data = caffe_model['fire7/expand1x1'].b.data
    dst.fire7.expand_33.W.data = caffe_model['fire7/expand3x3'].W.data
    dst.fire7.expand_33.b.data = caffe_model['fire7/expand3x3'].b.data
    print('copy fire7')

    dst.fire8.squeeze_11.W.data = caffe_model['fire8/squeeze1x1'].W.data
    dst.fire8.squeeze_11.b.data = caffe_model['fire8/squeeze1x1'].b.data
    dst.fire8.expand_11.W.data = caffe_model['fire8/expand1x1'].W.data
    dst.fire8.expand_11.b.data = caffe_model['fire8/expand1x1'].b.data
    dst.fire8.expand_33.W.data = caffe_model['fire8/expand3x3'].W.data
    dst.fire8.expand_33.b.data = caffe_model['fire8/expand3x3'].b.data
    print('copy fire8')

    dst.fire9.squeeze_11.W.data = caffe_model['fire9/squeeze1x1'].W.data
    dst.fire9.squeeze_11.b.data = caffe_model['fire9/squeeze1x1'].b.data
    dst.fire9.expand_11.W.data = caffe_model['fire9/expand1x1'].W.data
    dst.fire9.expand_11.b.data = caffe_model['fire9/expand1x1'].b.data
    dst.fire9.expand_33.W.data = caffe_model['fire9/expand3x3'].W.data
    dst.fire9.expand_33.b.data = caffe_model['fire9/expand3x3'].b.data
    print('copy fire9')

    dst.conv10.W.data = caffe_model['conv10'].W.data
    dst.conv10.b.data = caffe_model['conv10'].b.data
    print('copy conv10')

    print('done')


def _make_npz(path_npz, url, model):
    path_caffemodel = download.cached_download(url)
    sys.stderr.write(
        'Now loading caffemodel (usually it may take few minutes)\n')
    sys.stderr.flush()
    SqueezeNet.convert_caffemodel_to_npz(path_caffemodel, path_npz)
    npz.load_npz(path_npz, model)
    return model


def _retrieve(name_npz, url, model):
    root = download.get_dataset_directory('pfnet/chainer/models/')
    path = os.path.join(root, name_npz)
    return download.cache_or_load_file(
        path, lambda path: _make_npz(path, url, model),
        lambda path: npz.load_npz(path, model))
