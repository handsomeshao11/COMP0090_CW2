# Script to produce the visual mask predictions

import tensorflow as tf
import matplotlib.pyplot as plt
import time

import os
import sys


# add package to sys.path if it's not there already
# - so can import from dlCWK2TL package
try:
    # python package (dlCWK2TL) location - one level up from this file
    src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if src_path not in sys.path:
        sys.path.extend([src_path])
except NameError:
    print('issue with adding to path, probably due to __file__ not being defined')
    src_path = None

from dlCWK2TL import get_path
from dlCWK2TL.loader import H5ImageLoader
from dlCWK2TL.Models import *



if __name__ == "__main__":

    train_path = r'data_restructured/train/'
    val_path = r'data_restructured/val/'
    test_path = r'data_restructured/test/'

    img_path = r'images.h5'
    bin_path = r'binary.h5'
    bbox_path = r'bboxes.h5'
    mask_path = r'masks.h5' 

    if os.path.exists(get_path('visual_predictions')) is False:
        os.makedirs(get_path('visual_predictions'))

    unet_model_path = os.path.join(get_path('models'), 'unet_nontrained_ft_bestModel')
    vunet_model_path = os.path.join(get_path('models'), 'vunet_pretrain_non-ft_bestModel')
    vunet_ablated_model_path = os.path.join(get_path('models'), 'vunet_ablated_pretrain_ft_bestModel')
    runet_model_path = os.path.join(get_path('models'), 'runet_pretrain_non-ft_bestModel')
    dunet_model_path = os.path.join(get_path('models'), 'dunet_pretrain_non-ft_bestModel')
    munet_model_path = os.path.join(get_path('models'), 'munet_pretrain_non-ft_bestModel')

    test_gen = H5ImageLoader(test_path+img_path, test_path+mask_path, batch_size=20, type='segment')

    unet = tf.keras.models.load_model(unet_model_path, 
                                      custom_objects={'dice_coef':dice_coef, 'iou':iou, 'dice_coef_loss':dice_coef_loss})
    vunet = tf.keras.models.load_model(vunet_model_path, 
                                       custom_objects={'dice_coef':dice_coef, 'iou':iou, 'dice_coef_loss':dice_coef_loss})
    vunet_ablated = tf.keras.models.load_model(vunet_ablated_model_path, 
                                               custom_objects={'dice_coef':dice_coef, 'iou':iou, 'dice_coef_loss':dice_coef_loss})
    runet = tf.keras.models.load_model(runet_model_path, 
                                       custom_objects={'dice_coef':dice_coef, 'iou':iou, 'dice_coef_loss':dice_coef_loss})
    dunet = tf.keras.models.load_model(dunet_model_path, 
                                       custom_objects={'dice_coef':dice_coef, 'iou':iou, 'dice_coef_loss':dice_coef_loss})
    munet = tf.keras.models.load_model(munet_model_path, 
                                       custom_objects={'dice_coef':dice_coef, 'iou':iou, 'dice_coef_loss':dice_coef_loss})

    test_gen.__iter__()
    img, target = test_gen.__getitem__(0)

    unet_pred = unet.predict(img, verbose=0)[0]
    vunet_pred = vunet.predict(img, verbose=0)[0]
    vunet_ablated_pred = vunet_ablated.predict(img, verbose=0)[0]
    runet_pred = runet.predict(img, verbose=0)[0]
    dunet_pred = dunet.predict(img, verbose=0)[0]
    munet_pred = munet.predict(img, verbose=0)[0]

    fig, axes = plt.subplots(2, 4)
    axes[0, 0].imshow(img[0])
    axes[0, 1].imshow(target[0])
    axes[0, 2].imshow(unet_pred)
    axes[0, 3].imshow(vunet_pred)
    axes[1, 0].imshow(vunet_ablated_pred)
    axes[1, 1].imshow(runet_pred)
    axes[1, 2].imshow(dunet_pred)
    axes[1, 3].imshow(munet_pred)
    axes[0, 0].title.set_text('Original')
    axes[0, 1].title.set_text('Ground Truth')
    axes[0, 2].title.set_text('Baseline')
    axes[0, 3].title.set_text('V-Unet')
    axes[1, 0].title.set_text('V-Unet(ablated)')
    axes[1, 1].title.set_text('R-Unet')
    axes[1, 2].title.set_text('D-Unet')
    axes[1, 3].title.set_text('M-Unet')
    axes[0, 0].axis('off')
    axes[0, 1].axis('off')
    axes[0, 2].axis('off')
    axes[0, 3].axis('off')
    axes[1, 0].axis('off')
    axes[1, 1].axis('off')
    axes[1, 2].axis('off')
    axes[1, 3].axis('off')
    plt.tight_layout()
    fig.savefig(os.path.join(get_path('visual_predictions'),'{}.png'.format(time.time())))
    plt.show()

    fig, axes = plt.subplots(1, 5)
    axes[0].imshow(img[0])
    axes[1].imshow(target[0])
    axes[2].imshow(unet_pred > .5)
    axes[3].imshow(vunet_pred > .5)
    axes[4].imshow(vunet_ablated_pred > .5)
    axes[0].title.set_text('Original')
    axes[1].title.set_text('Ground Truth')
    axes[2].title.set_text('Baseline')
    axes[3].title.set_text('V-Unet')
    axes[4].title.set_text('V-Unet(ablated)')
    axes[0].axis('off')
    axes[1].axis('off')
    axes[2].axis('off')
    axes[3].axis('off')
    axes[4].axis('off')
    plt.tight_layout()
    fig.savefig(os.path.join(get_path('visual_predictions'),'mrp_{}.png'.format(time.time())))
    plt.show()

    fig, axes = plt.subplots(1, 5)
    axes[0].imshow(img[0])
    axes[1].imshow(target[0])
    axes[2].imshow(unet_pred)
    axes[3].imshow(vunet_pred)
    axes[4].imshow(vunet_ablated_pred)
    axes[0].title.set_text('Original')
    axes[1].title.set_text('Ground Truth')
    axes[2].title.set_text('Baseline')
    axes[3].title.set_text('V-Unet')
    axes[4].title.set_text('V-Unet(ablated)')
    axes[0].axis('off')
    axes[1].axis('off')
    axes[2].axis('off')
    axes[3].axis('off')
    axes[4].axis('off')
    plt.tight_layout()
    fig.savefig(os.path.join(get_path('visual_predictions'),'mrp_rawdata_{}.png'.format(time.time())))
    plt.show()

    fig, axes = plt.subplots(1, 6)
    axes[0].imshow(img[0])
    axes[1].imshow(target[0])
    axes[2].imshow(vunet_pred > .5)
    axes[3].imshow(runet_pred > .5)
    axes[4].imshow(dunet_pred > .5)
    axes[5].imshow(munet_pred > .5)
    axes[0].title.set_text('Original')
    axes[1].title.set_text('Ground Truth')
    axes[2].title.set_text('V-Unet')
    axes[3].title.set_text('R-Unet')
    axes[4].title.set_text('D-Unet')
    axes[5].title.set_text('M-Unet')
    axes[0].axis('off')
    axes[1].axis('off')
    axes[2].axis('off')
    axes[3].axis('off')
    axes[4].axis('off')
    axes[5].axis('off')
    plt.tight_layout()
    fig.savefig(os.path.join(get_path('visual_predictions'),'oeq_{}.png'.format(time.time())))
    plt.show()

    fig, axes = plt.subplots(1, 6)
    axes[0].imshow(img[0])
    axes[1].imshow(target[0])
    axes[2].imshow(vunet_pred)
    axes[3].imshow(runet_pred)
    axes[4].imshow(dunet_pred)
    axes[5].imshow(munet_pred)
    axes[0].title.set_text('Original')
    axes[1].title.set_text('Ground Truth')
    axes[2].title.set_text('V-Unet')
    axes[3].title.set_text('R-Unet')
    axes[4].title.set_text('D-Unet')
    axes[5].title.set_text('M-Unet')
    axes[0].axis('off')
    axes[1].axis('off')
    axes[2].axis('off')
    axes[3].axis('off')
    axes[4].axis('off')
    axes[5].axis('off')
    plt.tight_layout()
    fig.savefig(os.path.join(get_path('visual_predictions'),'oeq_rawdata_{}.png'.format(time.time())))
    plt.show()
    