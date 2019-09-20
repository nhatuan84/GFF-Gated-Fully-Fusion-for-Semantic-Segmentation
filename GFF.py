import tensorflow as tf
from tensorflow.contrib import slim
import numpy as np
from builders import frontend_builder
import os, sys

def Upsampling(inputs,feature_map_shape):
    return tf.image.resize_bilinear(inputs, size=feature_map_shape)

def ConvBlock(inputs, n_filters, kernel_size=[3, 3], stride=1):
    """
    Basic conv block for Encoder-Decoder
    Apply successivly Convolution, BatchNormalization, ReLU nonlinearity
    """
    net = tf.nn.relu(slim.batch_norm(inputs, fused=True))
    net = slim.conv2d(net, n_filters, kernel_size, stride=stride, activation_fn=None, normalizer_fn=None)
    return net

def FuseGFFConvBlock(inputs, n_filters, kernel_size=[3, 3], stride=1):
    """
    Basic conv block for Encoder-Decoder
    Apply successivly Convolution, BatchNormalization, ReLU nonlinearity
    """
    net = slim.conv2d(inputs, n_filters, kernel_size, stride=stride, activation_fn=None, normalizer_fn=None)
    net = tf.nn.relu(slim.batch_norm(net, fused=True))
    net = slim.conv2d(inputs, n_filters, kernel_size, stride=stride, activation_fn=None, normalizer_fn=None)
    net = tf.nn.relu(slim.batch_norm(net, fused=True))
    return net

def ResNetBlock_1(inputs, filters_1, filters_2):
    net = tf.nn.relu(slim.batch_norm(inputs, fused=True))
    net = slim.conv2d(net, filters_1, [1, 1], activation_fn=None, normalizer_fn=None)

    net = tf.nn.relu(slim.batch_norm(net, fused=True))
    net = slim.conv2d(net, filters_1, [3, 3], activation_fn=None, normalizer_fn=None)

    net = tf.nn.relu(slim.batch_norm(net, fused=True))
    net = slim.conv2d(net, filters_2, [1, 1], activation_fn=None, normalizer_fn=None)

    net = tf.add(inputs, net)

    return net

def ResNetBlock_2(inputs, filters_1, filters_2, s=1, d=1):
    net_1 = tf.nn.relu(slim.batch_norm(inputs, fused=True))
    net_1 = slim.conv2d(net_1, filters_1, [1, 1], stride=s, activation_fn=None, normalizer_fn=None)

    net_1 = tf.nn.relu(slim.batch_norm(net_1, fused=True))
    net_1 = slim.conv2d(net_1, filters_1, [3, 3], activation_fn=None, normalizer_fn=None)

    net_1 = tf.nn.relu(slim.batch_norm(net_1, fused=True))
    net_1 = slim.conv2d(net_1, filters_2, [1, 1], activation_fn=None, normalizer_fn=None)

    net_2 = tf.nn.relu(slim.batch_norm(inputs, fused=True))
    net_2 = slim.conv2d(net_2, filters_2, [1, 1], stride=s, rate=d, activation_fn=None, normalizer_fn=None)

    net = tf.add(net_1, net_2)

    return net


def ConvUpscaleBlock(inputs, n_filters, kernel_size=[3, 3], scale=2):
    """
    Basic conv transpose block for Encoder-Decoder upsampling
    Apply successivly Transposed Convolution, BatchNormalization, ReLU nonlinearity
    """
    net = tf.nn.relu(slim.batch_norm(inputs, fused=True))
    net = slim.conv2d_transpose(net, n_filters, kernel_size=[3, 3], stride=[scale, scale], activation_fn=None)
    return net

def InterpBlock(net, level, feature_map_shape, pooling_type):
    
    # Compute the kernel and stride sizes according to how large the final feature map will be
    # When the kernel size and strides are equal, then we can compute the final feature map size
    # by simply dividing the current size by the kernel or stride size
    # The final feature map sizes are 1x1, 2x2, 3x3, and 6x6. We round to the closest integer
    kernel_size = [int(np.round(float(feature_map_shape[0]) / float(level))), int(np.round(float(feature_map_shape[1]) / float(level)))]
    stride_size = kernel_size

    net = slim.pool(net, kernel_size, stride=stride_size, pooling_type='MAX')
    net = slim.conv2d(net, 512, [1, 1], activation_fn=None)
    net = slim.batch_norm(net, fused=True)
    net = tf.nn.relu(net)
    net = Upsampling(net, feature_map_shape)
    return net

def PyramidPoolingModule(inputs, feature_map_shape, pooling_type):
    """
    Build the Pyramid Pooling Module.
    """

    interp_block1 = InterpBlock(inputs, 1, feature_map_shape, pooling_type)
    interp_block2 = InterpBlock(inputs, 2, feature_map_shape, pooling_type)
    interp_block3 = InterpBlock(inputs, 3, feature_map_shape, pooling_type)
    interp_block6 = InterpBlock(inputs, 6, feature_map_shape, pooling_type)

    res = tf.concat([inputs, interp_block6, interp_block3, interp_block2, interp_block1], axis=-1)
    return res



def build_gffnet(inputs, label_size, num_classes, preset_model='PSPNet', frontend="ResNet101", pooling_type = "MAX",
    weight_decay=1e-5, upscaling_method="conv", is_training=True, pretrained_dir="models"):

    init_fn = None
    inputs = tf.identity(inputs, name='ymh_in')

    net = ConvBlock(inputs, n_filters=64, kernel_size=[3, 3])
    net = ConvBlock(net, n_filters=64, kernel_size=[7, 7], stride=2)
    net = slim.pool(net, [2, 2], stride=[2, 2], pooling_type='MAX')

    net = ResNetBlock_2(net, filters_1=64, filters_2=256, s=1)
    net = ResNetBlock_1(net, filters_1=64, filters_2=256)
    x1 = ResNetBlock_1(net, filters_1=64, filters_2=256) #(1, 80, 120, 256)
    x1n = slim.conv2d(x1, 256, [1, 1], stride=2, activation_fn=None, normalizer_fn=None)
    g1 = slim.conv2d(x1n, 256, [1, 1], activation_fn=None, normalizer_fn=None)
    g1 = tf.nn.sigmoid(g1)

    net = ResNetBlock_2(x1, filters_1=128, filters_2=512, s=2)
    net = ResNetBlock_1(net, filters_1=128, filters_2=512)
    net = ResNetBlock_1(net, filters_1=128, filters_2=512)
    x2 = ResNetBlock_1(net, filters_1=128, filters_2=512) #(1, 40, 60, 512)
    x2n = slim.conv2d(x2, 256, [1, 1], activation_fn=None, normalizer_fn=None)
    g2 = slim.conv2d(x2n, 256, [1, 1], activation_fn=None, normalizer_fn=None)
    g2 = tf.nn.sigmoid(g2)

    net = ResNetBlock_2(x2, filters_1=256, filters_2=1024, d=2)
    net = ResNetBlock_1(net, filters_1=256, filters_2=1024)
    net = ResNetBlock_1(net, filters_1=256, filters_2=1024)
    net = ResNetBlock_1(net, filters_1=256, filters_2=1024)
    net = ResNetBlock_1(net, filters_1=256, filters_2=1024)
    x3 = ResNetBlock_1(net, filters_1=256, filters_2=1024) #(1, 40, 60, 1024)
    x3n = slim.conv2d(x3, 256, [1, 1], activation_fn=None, normalizer_fn=None)
    g3 = slim.conv2d(x3n, 256, [1, 1], activation_fn=None, normalizer_fn=None)
    g3 = tf.nn.sigmoid(g3)

    net = ResNetBlock_2(x3, filters_1=512, filters_2=2048, d=4)
    net = ResNetBlock_1(net, filters_1=512, filters_2=2048)
    x4 = ResNetBlock_1(net, filters_1=512, filters_2=2048) #(1, 40, 60, 2048)
    x4n = slim.conv2d(x4, 256, [1, 1], activation_fn=None, normalizer_fn=None)
    g4 = slim.conv2d(x4n, 256, [1, 1], activation_fn=None, normalizer_fn=None)
    g4 = tf.nn.sigmoid(g4)

    x1gff = (1+g1)*x1n + (1-g1)*(g2*x2n + g3*x3n + g4*x4n)
    x2gff = (1+g2)*x2n + (1-g2)*(g1*x1n + g3*x3n + g4*x4n)
    x3gff = (1+g3)*x3n + (1-g3)*(g2*x2n + g1*x1n + g4*x4n)
    x4gff = (1+g4)*x4n + (1-g4)*(g2*x2n + g3*x3n + g1*x1n)

    x1gff = FuseGFFConvBlock(x1gff, 256)
    x2gff = FuseGFFConvBlock(x2gff, 256)
    x3gff = FuseGFFConvBlock(x3gff, 256)
    x4gff = FuseGFFConvBlock(x4gff, 256)

    feature_map_shape = [int(x / 8.0) for x in label_size]

    psp = PyramidPoolingModule(x4, feature_map_shape=feature_map_shape, pooling_type=pooling_type)

    d5 = tf.concat([psp, x1gff, x2gff, x3gff, x4gff], axis=-1)
    d4 = tf.concat([x1gff, x2gff, x3gff, x4gff], axis=-1)
    d3 = tf.concat([x1gff, x2gff, x3gff], axis=-1)
    d2 = tf.concat([x1gff, x2gff], axis=-1)
    d1 = tf.concat([x1gff], axis=-1)

    full_block = tf.concat([d1, d2, d3, d4, d5], axis=-1)

    net = slim.conv2d(full_block, 512, [3, 3], activation_fn=None)
    net = slim.batch_norm(net, fused=True)
    net = tf.nn.relu(net)

    if upscaling_method.lower() == "conv":
        net = ConvUpscaleBlock(net, 256, kernel_size=[3, 3], scale=2)
        net = ConvBlock(net, 256)
        net = ConvUpscaleBlock(net, 128, kernel_size=[3, 3], scale=2)
        net = ConvBlock(net, 128)
        net = ConvUpscaleBlock(net, 64, kernel_size=[3, 3], scale=2)
        net = ConvBlock(net, 64)
    elif upscaling_method.lower() == "bilinear":
        net = Upsampling(net, label_size)
    
    net = slim.conv2d(net, num_classes, [1, 1], activation_fn=None, scope='logits')
    net = tf.identity(net, name='ymh_out')

    #calculate aux loss
    if upscaling_method.lower() == "conv":
        loss = ConvUpscaleBlock(x3, 256, kernel_size=[3, 3], scale=2)
        loss = ConvBlock(loss, 256)
        loss = ConvUpscaleBlock(loss, 128, kernel_size=[3, 3], scale=2)
        loss = ConvBlock(loss, 128)
        loss = ConvUpscaleBlock(loss, 64, kernel_size=[3, 3], scale=2)
        loss = ConvBlock(loss, 64)
    elif upscaling_method.lower() == "bilinear":
        loss = Upsampling(x3, label_size)
    
    aux_loss = slim.conv2d(loss, num_classes, [1, 1], activation_fn=None, scope='aux_logits')

    return net, init_fn, aux_loss

