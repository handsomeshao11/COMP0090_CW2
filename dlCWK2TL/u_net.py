# Task script for the U-Net model

import tensorflow as tf
import pickle
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger
from tensorflow.keras.optimizers import Adam

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
    
    ### Build and fit the U-Net model, with parameters fixed after pre-training

    # Set up the data loading generator
    train_path = r'data_restructured/train/'
    val_path = r'data_restructured/val/'
    test_path = r'data_restructured/test/'

    img_path = r'images.h5'
    mask_path = r'masks.h5' 

    train_gen = H5ImageLoader(train_path+img_path, train_path+mask_path, batch_size=32, type='segment')
    val_gen = H5ImageLoader(val_path+img_path, val_path+mask_path, batch_size=20, type='segment')
    test_gen = H5ImageLoader(test_path+img_path, test_path+mask_path, batch_size=20, type='segment')

    img_shape = train_gen.img_shape
    n_epochs = 50


    # Set up the U-Net model
    model_name = 'runet'
    setting = ['not-pretrained', 'ft']
    model = unet_model(img_shape)
    model.compile(optimizer=Adam(lr=1e-5), 
                  loss=dice_coef_loss, 
                  metrics=[iou, dice_coef, 'binary_accuracy'])

    # Add models folder to save models, if folder does not already exist
    if os.path.exists(get_path('models')) is False:
        os.makedirs(get_path('models'))

    # Set up the checkpoint to save the best model
    checkpoint_path = os.path.join(get_path('models'), 
                                 '{}_{}_{}_bestModel'.format(model_name, setting[0], setting[1]))
    model_checkpoint = ModelCheckpoint(checkpoint_path,  
                                       verbose=1,
                                       monitor='val_loss',
                                       save_best_only=True)
    
    # Set up a CSV log path to save the evaluation results per epoch
    CSVLog_path = os.path.join(get_path('evaluation_results'), 
                                 '{}_{}_{}.csv'.format(model_name, setting[0], setting[1]))
    pretrain_csvlogger = CSVLogger(filename=CSVLog_path , separator=",", append=True)
    callbacks_list = [model_checkpoint, pretrain_csvlogger]

    # Fit the model and save it
    history = model.fit(train_gen,
                        steps_per_epoch=train_gen.num_images//train_gen.batch_size, 
                        epochs=n_epochs,
                        callbacks=callbacks_list, 
                        validation_data=val_gen,
                        validation_steps=val_gen.num_images//val_gen.batch_size,
                        verbose=1)
    model.save(os.path.join(get_path('models'), 
                            '{}_{}_{}_model'.format(model_name, setting[0], setting[1])))

    # evaluate the model with test data
    result = model.evaluate(test_gen, 
                            steps=test_gen.num_images//test_gen.batch_size, 
                            verbose=0)
    results = dict(zip(model.metrics_names,result))
    
    # Save the test data evaluation results
    test_eval_path = os.path.join(get_path('evaluation_results'), 
                                  '{}_{}_{}_eval.txt'.format(model_name, setting[0], setting[1]))
    with open(test_eval_path, 'w') as f:
        for k in results.keys():
            f.write(k + ': ' + str(results[k]) + '\n')

    # plot the results
    keys = history.history.keys()
    fig, axs = plt.subplots(1, len(keys)//2, figsize = (25, 5))

    # Create file paths for the performance data and charts over all epochs
    history_path = os.path.join(get_path('evaluation_results'), 
                                '{}_{}_{}_history'.format(model_name, setting[0], setting[1]))    
    eval_chart_path = os.path.join(get_path('evaluation_results'), 
                                   '{}_{}_{}.png'.format(model_name, setting[0], setting[1]))

    # Compile the performance history of the training and validation data and save as a chart
    # and as a png chart
    for k, key in enumerate(list(keys)[:len(keys)//2]):
        training = history.history[key]
        validation = history.history['val_' + key]
        epoch_count = range(1, len(training) + 1)
        axs[k].plot(epoch_count, training, 'r--')
        axs[k].plot(epoch_count, validation, 'b-')
        axs[k].legend(['Training ' + key, 'Validation ' + key])
        axs[k].set_title(key)
    with open(history_path , 'wb') as file_pi:
        pickle.dump(history.history, file_pi)
    fig.savefig(eval_chart_path)
    plt.show()    
