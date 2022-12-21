import os
import numpy as np
import h5py

import tensorflow as tf
import tensorflow.keras
from tensorflow.keras import backend as K
from tensorflow.keras import Input

from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Flatten, Input, Activation, BatchNormalization, Dropout, Lambda, Conv2D, Conv2DTranspose, MaxPooling2D, concatenate, LeakyReLU
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger
from tensorflow.keras.optimizers import Adam

from tensorflow.keras import applications


# The v_unet_model, unet_model, dice_coef, dice_coef_loss and iou are functions from the 2021
# research paper "Transfer Learning U-Net Deep Learning for Lung Ultrasound Segmentation".
# These functions are the work of the authors of this paper and can be found in their GitHub
# repository https://github.com/dorltcheng/Transfer-Learning-U-Net-Deep-Learning-for-Lung-Ultrasound-Segmentation

# For the models m_unet_model, d_unet_model and r_unet_model, we modified the v_unet_model
# from the above source by removing the VGG16 encoder and integrating a different network
# for each function. 
 
def v_unet_model(input_shape, pre_train='imagenet', fine_tuning=False):
    """U-Net architecture with a VGG16 encoder. The encoder can be pre trained and
    the parameters of the encoder can be fixed.
    
    input: input_shape (height, width, channels) 
    return model"""
    input_shape = input_shape
    base_VGG = applications.vgg16.VGG16(include_top=False, 
                                        weights=pre_train, 
                                        input_shape=input_shape)

    # freezing layers (if specified)
    if fine_tuning == False:
        for layer in base_VGG.layers: 
            layer.trainable = False

    # the bridge (exclude the last maxpooling layer in VGG16) 
    bridge = base_VGG.get_layer("block5_conv3").output
    print(bridge.shape)

    # Decoder now
    up1 = Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(bridge)
    print(up1.shape)
    concat_1 = concatenate([up1, base_VGG.get_layer("block4_conv3").output], axis=3)
    conv6 = Conv2D(512, (3, 3), activation='relu', padding='same')(concat_1)
    conv6 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv6)

    up2 = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv6)
    print(up2.shape)
    concat_2 = concatenate([up2, base_VGG.get_layer("block3_conv3").output], axis=3)
    conv7 = Conv2D(256, (3, 3), activation='relu', padding='same')(concat_2)
    conv7 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv7)

    up3 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv7)
    print(up3.shape)
    concat_3 = concatenate([up3, base_VGG.get_layer("block2_conv2").output], axis=3)
    conv8 = Conv2D(128, (3, 3), activation='relu', padding='same')(concat_3)
    conv8 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv8)

    up4 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv8)
    print(up4.shape)
    concat_4 = concatenate([up4, base_VGG.get_layer("block1_conv2").output], axis=3)
    conv9 = Conv2D(64, (3, 3), activation='relu', padding='same')(concat_4)
    conv9 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv9)

    conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)
    print(conv10.shape)

    model_ = Model(inputs=[base_VGG.input], outputs=[conv10])

    return model_
        

def r_unet_model(input_shape, pre_train='imagenet', fine_tuning=False):
    """U-Net architecture with a ResNet50 encoder. The encoder can be pre trained and
    the parameters of the encoder can be fixed.
    
    input: input_shape (height, width, channels) 
    return model"""
    input_shape = input_shape
    base_ResNet = applications.resnet50.ResNet50(include_top=False, 
                                                    weights=pre_train, 
                                                    input_shape=input_shape)

    # freezing layers (if specified)
    if fine_tuning == False:
        for layer in base_ResNet.layers: 
            layer.trainable = False

    # the bridge (exclude the last maxpooling layer) 
    # output: 8x8x1024
    bridge = base_ResNet.get_layer("conv5_block3_out").output
    print(bridge.shape)

    # Decoder now 
    # output: 16x16 (channels of ResNet encoder: 1024)
    up1 = Conv2DTranspose(1024, (2, 2), strides=(2, 2), padding='same')(bridge)
    print(up1.shape)
    concat_1 = concatenate([up1, base_ResNet.get_layer("conv4_block6_out").output], axis=3)
    conv6 = Conv2D(1024, (3, 3), activation='relu', padding='same')(concat_1)
    conv6 = Conv2D(1024, (3, 3), activation='relu', padding='same')(conv6)
    # output: 32x32 (channels of ResNet encoder: 512)
    up2 = Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(conv6)
    print(up2.shape)
    concat_2 = concatenate([up2, base_ResNet.get_layer("conv3_block4_out").output], axis=3)
    conv7 = Conv2D(512, (3, 3), activation='relu', padding='same')(concat_2)
    conv7 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv7)
    # output: 64x64 (channels of ResNet encoder: 256)
    up3 = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv7)
    print(up3.shape)
    concat_3 = concatenate([up3, base_ResNet.get_layer("conv2_block3_out").output], axis=3)
    conv8 = Conv2D(256, (3, 3), activation='relu', padding='same')(concat_3)
    conv8 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv8)
    # output: 128x128 (channels of ResNet encoder: 64)
    up4 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv8)
    print(up4.shape)
    concat_4 = concatenate([up4, base_ResNet.get_layer("conv1_relu").output], axis=3)
    conv9 = Conv2D(64, (3, 3), activation='relu', padding='same')(concat_4)
    conv9 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv9)
    # output: 256x256 (channels of ResNet encoder: 3)
    up5 = Conv2DTranspose(3, (2, 2), strides=(2, 2), padding='same')(conv9)
    print(up5.shape)
    concat_5 = concatenate([up5, base_ResNet.get_layer("input_1").output], axis=3)
    conv10 = Conv2D(3, (3, 3), activation='relu', padding='same')(concat_5)
    conv10 = Conv2D(3, (3, 3), activation='relu', padding='same')(conv10)
    
    conv11 = Conv2D(1, (1, 1), activation='sigmoid')(conv10)
    print(conv11.shape)

    model_ = Model(inputs=[base_ResNet.input], outputs=[conv11])
    
    return model_


def d_unet_model(input_shape, pre_train='imagenet', fine_tuning=False):
    """U-Net architecture with a DenseNet121 encoder. The encoder can be pre trained and
    the parameters of the encoder can be fixed.
    
    input: input_shape (height, width, channels) 
    return model"""
    input_shape = input_shape
    base_DeneseNet = applications.densenet.DenseNet121(include_top=False, 
                                                    weights=pre_train, 
                                                    input_shape=input_shape)

    # freezing layers (if specified)
    if fine_tuning == False:
        for layer in base_DeneseNet.layers: 
            layer.trainable = False

    # the bridge (exclude the last maxpooling layer) 
    # output: 8x8x1024
    bridge = base_DeneseNet.get_layer("relu").output
    print(bridge.shape)

    # Decoder now 
    # output: 16x16 (channels of DeneseNet encoder: 512)
    up1 = Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(bridge)
    print(up1.shape)
    concat_1 = concatenate([up1, base_DeneseNet.get_layer("pool4_conv").output], axis=3)
    conv6 = Conv2D(512, (3, 3), activation='relu', padding='same')(concat_1)
    conv6 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv6)
    # output: 32x32 (channels of DeneseNet encoder: 256)
    up2 = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv6)
    print(up2.shape)
    concat_2 = concatenate([up2, base_DeneseNet.get_layer("pool3_conv").output], axis=3)
    conv7 = Conv2D(256, (3, 3), activation='relu', padding='same')(concat_2)
    conv7 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv7)
    # output: 64x64 (channels of DeneseNet encoder: 128)
    up3 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv7)
    print(up3.shape)
    concat_3 = concatenate([up3, base_DeneseNet.get_layer("pool2_conv").output], axis=3)
    conv8 = Conv2D(128, (3, 3), activation='relu', padding='same')(concat_3)
    conv8 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv8)
    # output: 128x128 (channels of DeneseNet encoder: 64)
    up4 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv8)
    print(up4.shape)
    concat_4 = concatenate([up4, base_DeneseNet.get_layer("conv1/relu").output], axis=3)
    conv9 = Conv2D(64, (3, 3), activation='relu', padding='same')(concat_4)
    conv9 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv9)
    # output: 256x256 (channels of DeneseNet encoder: 3)
    up5 = Conv2DTranspose(3, (2, 2), strides=(2, 2), padding='same')(conv9)
    print(up5.shape)
    concat_5 = concatenate([up5, base_DeneseNet.get_layer("input_1").output], axis=3)
    conv10 = Conv2D(3, (3, 3), activation='relu', padding='same')(concat_5)
    conv10 = Conv2D(3, (3, 3), activation='relu', padding='same')(conv10)
    
    conv11 = Conv2D(1, (1, 1), activation='sigmoid')(conv10)
    print(conv11.shape)

    model_ = Model(inputs=[base_DeneseNet.input], outputs=[conv11])
    
    return model_       

def m_unet_model(input_shape, pre_train='imagenet', fine_tuning=False):
    """U-Net architecture with a MobileNetV2 encoder. The encoder can be pre trained and
    the parameters of the encoder can be fixed.
    
    input: input_shape (height, width, channels) 
    return model"""

    input_shape = input_shape
    base_MobielNet = applications.mobilenet_v2.MobileNetV2(include_top=False, 
                                                           weights=pre_train, 
                                                           input_shape=input_shape)
    # freezing layers (if specified)
    if fine_tuning == False:
        for layer in base_MobielNet.layers: 
            layer.trainable = False

    # the bridge (exclude the last maxpooling layer) 
    # output: 8x8x1024
    bridge = base_MobielNet.get_layer("out_relu").output
    print(bridge.shape)

    # Decoder now 
    # output: 16x16 (channels of DeneseNet encoder: 512)
    up1 = Conv2DTranspose(576, (2, 2), strides=(2, 2), padding='same')(bridge)
    print(up1.shape)
    concat_1 = concatenate([up1, base_MobielNet.get_layer("block_13_expand_relu").output], axis=3)
    conv6 = Conv2D(576, (3, 3), activation='relu', padding='same')(concat_1)
    conv6 = Conv2D(576, (3, 3), activation='relu', padding='same')(conv6)
    # output: 32x32 (channels of DeneseNet encoder: 256)
    up2 = Conv2DTranspose(196, (2, 2), strides=(2, 2), padding='same')(conv6)
    print(up2.shape)
    concat_2 = concatenate([up2, base_MobielNet.get_layer("block_6_expand_relu").output], axis=3)
    conv7 = Conv2D(196, (3, 3), activation='relu', padding='same')(concat_2)
    conv7 = Conv2D(196, (3, 3), activation='relu', padding='same')(conv7)
    # output: 64x64 (channels of DeneseNet encoder: 128)
    up3 = Conv2DTranspose(144, (2, 2), strides=(2, 2), padding='same')(conv7)
    print(up3.shape)
    concat_3 = concatenate([up3, base_MobielNet.get_layer("block_3_expand_relu").output], axis=3)
    conv8 = Conv2D(144, (3, 3), activation='relu', padding='same')(concat_3)
    conv8 = Conv2D(144, (3, 3), activation='relu', padding='same')(conv8)
    # output: 128x128 (channels of DeneseNet encoder: 64)
    up4 = Conv2DTranspose(96, (2, 2), strides=(2, 2), padding='same')(conv8)
    print(up4.shape)
    concat_4 = concatenate([up4, base_MobielNet.get_layer("block_1_expand_relu").output], axis=3)
    conv9 = Conv2D(96, (3, 3), activation='relu', padding='same')(concat_4)
    conv9 = Conv2D(96, (3, 3), activation='relu', padding='same')(conv9)
    # output: 256x256 (channels of DeneseNet encoder: 3)
    up5 = Conv2DTranspose(3, (2, 2), strides=(2, 2), padding='same')(conv9)
    print(up5.shape)
    concat_5 = concatenate([up5, base_MobielNet.get_layer("input_1").output], axis=3)
    conv10 = Conv2D(3, (3, 3), activation='relu', padding='same')(concat_5)
    conv10 = Conv2D(3, (3, 3), activation='relu', padding='same')(conv10)
    
    conv11 = Conv2D(1, (1, 1), activation='sigmoid')(conv10)
    print(conv11.shape)

    model_ = Model(inputs=[base_MobielNet.input], outputs=[conv11])

    return model_ 



def unet_model(input_shape):
    """Standard U-Net model"""
    
    inp = Input(input_shape)
    
    # contracting path 
    conv1 = Conv2D(64, (3, 3), activation='relu', padding='same')(inp)
    conv1 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool3)
    conv4 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(1024, (3, 3), activation='relu', padding='same')(pool4)
    conv5 = Conv2D(1024, (3, 3), activation='relu', padding='same')(conv5)

    # Expanding path 
    up1 = Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(conv5)
    print(up1.shape)
    
    concat_1 = concatenate([up1, conv4], axis=3)
    
    conv6 = Conv2D(512, (3, 3), activation='relu', padding='same')(concat_1)
    conv6 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv6)

    up2 = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv6)
    print(up2.shape)
    
    concat_2 = concatenate([up2, conv3], axis=3)
    
    conv7 = Conv2D(256, (3, 3), activation='relu', padding='same')(concat_2)
    conv7 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv7)

    up3 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv7)
    print(up3.shape)
    
    concat_3 = concatenate([up3, conv2], axis=3)
    
    conv8 = Conv2D(128, (3, 3), activation='relu', padding='same')(concat_3)
    conv8 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv8)

    up4 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv8)
    print(up4.shape)
    
    concat_4 = concatenate([up4, conv1], axis=3)
    
    conv9 = Conv2D(64, (3, 3), activation='relu', padding='same')(concat_4)
    conv9 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv9)

    conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)
    print(conv10.shape)

    model_ = Model(inputs=[inp], outputs=[conv10])
    
    return model_




def dice_coef(y_true, y_pred, smooth = 1.):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

def iou(y_true, y_pred, smooth = 1.):
    intersection = K.sum(y_true * y_pred)
    sum_ = K.sum(y_true) + K.sum(y_pred)
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return jac

