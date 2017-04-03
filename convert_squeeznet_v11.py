# -*- coding: utf-8 -*-
"""
Created on Fri Feb  3 11:25:39 2017

@author: yunosuke.ishikawa
"""

def convert_squeezenet_v11(src, dst):
    # copy parameters from caffemodel into chainer model
    print('start copy params.')
    caffe_model = src
	
    dst.conv1.W.data = caffe_model['conv1'].W.data
    dst.conv1.b.data = caffe_model['conv1'].b.data
    print('copy conv1')
    
    dst.fire2_squeeze_11.W.data = caffe_model['fire2/squeeze1x1'].W.data
    dst.fire2_squeeze_11.b.data = caffe_model['fire2/squeeze1x1'].b.data	
    dst.fire2_expand_11.W.data = caffe_model['fire2/expand1x1'].W.data
    dst.fire2_expand_11.b.data = caffe_model['fire2/expand1x1'].b.data
    dst.fire2_expand_33.W.data = caffe_model['fire2/expand3x3'].W.data
    dst.fire2_expand_33.b.data = caffe_model['fire2/expand3x3'].b.data
    print('copy fire2')
    
    dst.fire3_squeeze_11.W.data = caffe_model['fire3/squeeze1x1'].W.data
    dst.fire3_squeeze_11.b.data = caffe_model['fire3/squeeze1x1'].b.data	
    dst.fire3_expand_11.W.data = caffe_model['fire3/expand1x1'].W.data
    dst.fire3_expand_11.b.data = caffe_model['fire3/expand1x1'].b.data
    dst.fire3_expand_33.W.data = caffe_model['fire3/expand3x3'].W.data
    dst.fire3_expand_33.b.data = caffe_model['fire3/expand3x3'].b.data
    print('copy fire3')
	
    dst.fire4_squeeze_11.W.data = caffe_model['fire4/squeeze1x1'].W.data
    dst.fire4_squeeze_11.b.data = caffe_model['fire4/squeeze1x1'].b.data	
    dst.fire4_expand_11.W.data = caffe_model['fire4/expand1x1'].W.data
    dst.fire4_expand_11.b.data = caffe_model['fire4/expand1x1'].b.data
    dst.fire4_expand_33.W.data = caffe_model['fire4/expand3x3'].W.data
    dst.fire4_expand_33.b.data = caffe_model['fire4/expand3x3'].b.data
    print('copy fire4')
	
    dst.fire5_squeeze_11.W.data = caffe_model['fire5/squeeze1x1'].W.data
    dst.fire5_squeeze_11.b.data = caffe_model['fire5/squeeze1x1'].b.data	
    dst.fire5_expand_11.W.data = caffe_model['fire5/expand1x1'].W.data
    dst.fire5_expand_11.b.data = caffe_model['fire5/expand1x1'].b.data
    dst.fire5_expand_33.W.data = caffe_model['fire5/expand3x3'].W.data
    dst.fire5_expand_33.b.data = caffe_model['fire5/expand3x3'].b.data
    print('copy fire5')

    dst.fire6_squeeze_11.W.data = caffe_model['fire6/squeeze1x1'].W.data
    dst.fire6_squeeze_11.b.data = caffe_model['fire6/squeeze1x1'].b.data	
    dst.fire6_expand_11.W.data = caffe_model['fire6/expand1x1'].W.data
    dst.fire6_expand_11.b.data = caffe_model['fire6/expand1x1'].b.data
    dst.fire6_expand_33.W.data = caffe_model['fire6/expand3x3'].W.data
    dst.fire6_expand_33.b.data = caffe_model['fire6/expand3x3'].b.data
    print('copy fire6')
	
    dst.fire7_squeeze_11.W.data = caffe_model['fire7/squeeze1x1'].W.data
    dst.fire7_squeeze_11.b.data = caffe_model['fire7/squeeze1x1'].b.data	
    dst.fire7_expand_11.W.data = caffe_model['fire7/expand1x1'].W.data
    dst.fire7_expand_11.b.data = caffe_model['fire7/expand1x1'].b.data
    dst.fire7_expand_33.W.data = caffe_model['fire7/expand3x3'].W.data
    dst.fire7_expand_33.b.data = caffe_model['fire7/expand3x3'].b.data
    print('copy fire7')
	
    dst.fire8_squeeze_11.W.data = caffe_model['fire8/squeeze1x1'].W.data
    dst.fire8_squeeze_11.b.data = caffe_model['fire8/squeeze1x1'].b.data	
    dst.fire8_expand_11.W.data = caffe_model['fire8/expand1x1'].W.data
    dst.fire8_expand_11.b.data = caffe_model['fire8/expand1x1'].b.data
    dst.fire8_expand_33.W.data = caffe_model['fire8/expand3x3'].W.data
    dst.fire8_expand_33.b.data = caffe_model['fire8/expand3x3'].b.data
    print('copy fire8')
	
    dst.fire9_squeeze_11.W.data = caffe_model['fire9/squeeze1x1'].W.data
    dst.fire9_squeeze_11.b.data = caffe_model['fire9/squeeze1x1'].b.data	
    dst.fire9_expand_11.W.data = caffe_model['fire9/expand1x1'].W.data
    dst.fire9_expand_11.b.data = caffe_model['fire9/expand1x1'].b.data
    dst.fire9_expand_33.W.data = caffe_model['fire9/expand3x3'].W.data
    dst.fire9_expand_33.b.data = caffe_model['fire9/expand3x3'].b.data
    print('copy fire9')
	
    dst.conv10.W.data = caffe_model['conv10'].W.data
    dst.conv10.b.data = caffe_model['conv10'].b.data
    print('copy conv10')
        
    print('done')