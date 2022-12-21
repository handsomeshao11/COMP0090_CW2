# Data generator pipeline

import numpy as np
import random
import h5py
import matplotlib.pyplot as plt

import time
import tensorflow as tf


class H5ImageLoader(tf.keras.utils.Sequence):
    def __init__(self, img_file, target_file, batch_size=32, shuffle=True, type='bin_cls'):
        self.img_h5 = h5py.File(img_file,'r')
        self.dataset_list = list(self.img_h5.keys())
        self.target_h5 = h5py.File(target_file,'r')
        self.num_images = len(self.img_h5)
        self.batch_size = batch_size
        self.num_batches = int(self.num_images/self.batch_size)      
        self.img_ids = [i for i in range(self.num_images)] 
        self.img_shape = self.img_h5[list(self.img_h5.keys())[0]].shape
        self.target_shape = self.target_h5[list(self.target_h5.keys())[0]].shape
        self.shuffle = shuffle
        self.type = type
        if self.shuffle:
            random.shuffle(self.img_ids)
    
    def __len__(self):
        return self.num_batches
    
    def __getitem__(self, idx):
        batch_img_ids = self.img_ids[idx*self.batch_size:(idx+1)*self.batch_size]
        datasets = [self.dataset_list[i] for i in batch_img_ids]
        
        images = [self.img_h5[ds][()]/255. for ds in datasets]
        if self.type=='bin_cls':
            targets = [int(self.target_h5[ds][()]) for ds in datasets]
            targets = tf.one_hot(targets, depth=2)
        elif self.type=='bbox':
            targets = [self.target_h5[ds][()]/255. for ds in datasets]
        elif self.type=='segment':
            targets = [self.target_h5[ds][()] for ds in datasets] 
        return np.array(images), np.array(targets)

    def on_epoch_end(self):
        if self.shuffle:
            random.shuffle(self.img_ids)


# Testing code run in command line
if __name__ == '__main__':
    train_path = r'data_restructured/train/'
    val_path = r'data_restructured/val/'
    test_path = r'data_restructured/test/'

    img_path = r'images.h5'
    bin_path = r'binary.h5'
    bbox_path = r'bboxes.h5'
    mask_path = r'masks.h5' 
    
    bin_generators, box_generators, seg_generators = [], [], []
    for dir in [train_path, val_path, test_path]:
        bin_generators.append(H5ImageLoader(dir+img_path, dir+bin_path, batch_size=1, type='bin_cls'))
        box_generators.append(H5ImageLoader(dir+img_path, dir+bbox_path, batch_size=1,type='bbox'))
        seg_generators.append(H5ImageLoader(dir+img_path, dir+mask_path, batch_size=1,type='segment'))

    start = time.time()
    bin_images, bin_targets = [[],[],[]], [[],[],[]]
    box_images, box_targets = [[],[],[]], [[],[],[]]
    seg_images, seg_targets = [[],[],[]], [[],[],[]]   
    for i, t in bin_generators[0]:
        bin_images[0].extend(i)
        bin_targets[0].extend(t)
    for i, t in box_generators[0]:
        box_images[0].extend(i)
        box_targets[0].extend(t)
    for i, t in seg_generators[0]:
        seg_images[0].extend(i)
        seg_targets[0].extend(t)
    print('Train data are loaded...')
    for i, t in bin_generators[1]:
        bin_images[1].extend(i)
        bin_targets[1].extend(t)
    for i, t in box_generators[1]:
        box_images[1].extend(i)
        box_targets[1].extend(t)
    for i, t in seg_generators[1]:
        seg_images[1].extend(i)
        seg_targets[1].extend(t)
    print('Val data are loaded...')
    for i, t in bin_generators[2]:
        bin_images[2].extend(i)
        bin_targets[2].extend(t)
    for i, t in box_generators[2]:
        box_images[2].extend(i)
        box_targets[2].extend(t)
    for i, t in seg_generators[2]:
        seg_images[2].extend(i)
        seg_targets[2].extend(t)
    print('Test data are loaded...')
    print('')
    print('Iteration time: ', time.time()-start)
    print('')
    
    cls = ['Train', 'Val', 'Test']
    for i in range(3):
        print(cls[i]+' data:')
        print(' Binary classification')
        print('  # of samples (images, targets):{}, {}'.format(len(bin_images[i]), len(bin_targets[i])))
        print('  Shape (images, targets):{}, {}'.format(bin_images[i][0].shape, bin_targets[i][0].shape))
        print(' Bounding box')
        print('  # of samples (images, targets):{}, {}'.format(len(box_images[i]), len(box_targets[i])))
        print('  Shape (images, targets):{}, {}'.format(box_images[i][0].shape, box_targets[i][0].shape))
        print(' Segmentation')
        print('  # of samples (images, targets):{}, {}'.format(len(seg_images[i]), len(seg_targets[i])))
        print('  Shape (images, targets):{}, {}'.format(seg_images[i][0].shape, seg_targets[i][0].shape))
    print('')
    
    datasets =  [bin_images, bin_targets, box_images, box_targets, seg_images, seg_targets]
    datasets = [bin_generators, box_generators, seg_generators]
    datasets_name =  ['bin_generators', 'box_generators', 'seg_generators']
    for i, ds in enumerate(datasets):
        for j in range(3):    
            index = []
            for k in range(len(ds)):
                if ds[k]==None:
                    index.append(k)
            if len(index)>0:
                print(datasets_name[i] + '({})'.format(cls[j]) + ': Missing target values: ' + index)
            else:
                print(datasets_name[i] + '({})'.format(cls[j]) + ': No missing target values')
    print('')
    
    print('Numbers of each binary class:')
    for i, ds in enumerate(bin_targets):
        cats=0
        dogs=0
        other=0
        for j in range(len(ds)):
            if ds[j][0]==1 and ds[j][1]==0:
                cats+=1
            elif ds[j][0]==0 and ds[j][1]==1:
                dogs+=1
            else:
                other+=1
        print(' ' + cls[i])
        print('  Cats:'+str(cats) + ', Dogs:'+str(dogs) + ', Other:'+str(other))
    print('')
      
    num_images=10
    print(bin_targets[0][:num_images])
    _, axes = plt.subplots(1, num_images)
    axes = axes.flatten()
    for img, ax in zip(bin_images[0][:num_images], axes):
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.show()
    
    num_images=10
    _, axes = plt.subplots(1, num_images)
    axes = axes.flatten()
    for img, ax in zip(seg_images[0][:num_images], axes):
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.show() 
    
    num_images=10
    _, axes = plt.subplots(1, num_images)
    axes = axes.flatten()
    for img, ax in zip(seg_targets[0][:num_images], axes):
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.show()
    

