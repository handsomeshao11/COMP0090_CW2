import os
import requests
import tarfile
import shutil
import h5py


DATA_PATH = './data'
dir_names = ['train', 'val', 'test']
file_names = ['images.h5', 'binary.h5', 'bboxes.h5', 'masks.h5']
url_base = 'https://weisslab.cs.ucl.ac.uk/WEISSTeaching/datasets/-/archive/oxpet/datasets-oxpet.tar.gz?path=data_new/'

# Refrech directories
if os.path.exists(DATA_PATH):
    shutil.rmtree(DATA_PATH)
os.makedirs(DATA_PATH)

# Download the files and store them in the directories
print('Downloading and extracting data...')
for dir in dir_names:
    url = url_base + dir
    print(' '+ dir + ' data downloading...')
    r = requests.get(url, allow_redirects=True)
    _ = open(dir,'wb').write(r.content)
    with tarfile.open(dir) as tar_obj:
        tar_obj.extractall(DATA_PATH)
        tar_obj.close()
    os.remove(dir)
    shutil.move(DATA_PATH+'/datasets-oxpet-data_new-{}/data_new/{}'.format(dir,dir), DATA_PATH+'/'+dir)
    shutil.rmtree(DATA_PATH+'/datasets-oxpet-data_new-{}'.format(dir))

# Restructure the h5 files 
new_path = './data_restructured'
if os.path.exists(new_path):
    shutil.rmtree(new_path)
os.makedirs(new_path)
for dir in dir_names:
    os.makedirs(new_path+'/'+dir)
    for file in file_names:
        with h5py.File(DATA_PATH+'/'+dir+'/'+file, 'r') as h5_r:
            with h5py.File(new_path+'/'+dir+'/'+file, 'w') as h5_w:
                for i in range(len(h5_r[list(h5_r.keys())[0]])):
                    h5_w.create_dataset(str(i), data=h5_r[list(h5_r.keys())[0]][i])
                print(new_path+'/'+dir+'/'+file+' is Done.')
