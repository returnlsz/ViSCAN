# This script is used to generate dataset indices for training and testing: train_seq and val_seq，the two files are used for ViSCAN training
# usage: python generate_npy.py
# PS:For a specific fold, you need to adjust the value of k at the end of the script in the line for k, (Trindex, Tsindex) in enumerate(folder.split(all_files)):.
# Similarly, you need to modify the path: paths.append(os.path.join(dirname, filename).replace('\\', '/').replace('autodl-tmp/YOLOV/', '')).
import os
import numpy as np
from sklearn.model_selection import KFold
 
# Obtain the label to check whether the data exists. Some data exists, but no label is found
list_train=[]
list_val=[]
path1 = r"ILSVRC2015/Annotations/VID/train/ILSVRC2015_VID_train_0000"
path2 = r"ILSVRC2015/Annotations/VID/train/ILSVRC2015_VID_train_0000_augmented"
# path3 = r"ILSVRC2015/Annotations/VID/train/NEA_DATA"
# path2 = r"D:\Aware_model\ILSVRC2015\Annotations\VID\train\ILSVRC2015_VID_train_0001"
# path3 = r"D:\Aware_model\ILSVRC2015\Annotations\VID\train\ILSVRC2015_VID_train_0002"
# path4 = r"D:\Aware_model\ILSVRC2015\Annotations\VID\train\ILSVRC2015_VID_train_0003"
path_val = r'ILSVRC2015/Annotations/VID/val'
path_val2 = r'ILSVRC2015/Annotations/VID/val_augmented'

f1 = os.listdir(path1)  # Get the names of all folders under ILSVRC2015_VID_train_0000
f2 = os.listdir(path2)
# f3 = os.listdir(path3)
# f2 = os.listdir(path2)
# f3 = os.listdir(path3)
# f4 = os.listdir(path4)
f_val = os.listdir(path_val)
f_val2 = os.listdir(path_val2)
list_train.append(f1)
list_train.append(f2)
# list_train.append(f3)
# list_train.append(f2)
# list_train.append(f3)
# list_train.append(f4)
list_val.append(f_val)
list_val.append(f_val2)
 
list_train = [item for list in list_train for item in list]  
list_val = [item for list in list_val for item in list]
print('trian',list_train)
print('val',list_val)
 
all_files = []

# Obtain the path of file data to form an npy file
def get_path(id,path,sava_path):
    paths2=[]
    if id=='val': #path:'D:\Aware_model\ILSVRC2015\Data\VID\val'
        for dirname, _, filenames in os.walk(path):  # dirname:D:\Aware_model\ILSVRC2015\Data\VID\val\ILSVRC2015_val_00000000
            # print(dirname," ......" ,filenames)      # filenames:['000000.JPEG', '000001.JPEG', '000002.JPEG', '000003.JPEG', '000004.JPEG'...]
            name = dirname.split('/')[-1]           #ILSVRC2015_val_00000000
            # print(name)
            if name in list_val:  
                # print(name)
                paths = []
                for filename in filenames:
                    paths.append(os.path.join(dirname, filename).replace('\\','/').replace('autodl-tmp/ViSCAN/','')) 
                # print(paths)
                if len(paths) >0:
                    all_files.append(paths)
                    #paths2.append(paths)
            #np.save(sava_path, paths2, allow_pickle=True, fix_imports=True)
    else:
        for dirname, _, filenames in os.walk(path):
            name = dirname.split('/')[-1]
            # print(name,list_train)
            if name in list_train:
                print(name)
                paths = []
                for filename in filenames:
                    paths.append(os.path.join(dirname, filename).replace('\\', '/').replace('autodl-tmp/ViSCAN/', ''))
                # print(paths)
                if len(paths) > 0:
                    all_files.append(paths)
                    # paths2.append(paths)
            #np.save(sava_path, paths2, allow_pickle=True, fix_imports=True)
 
get_path('train',r'ILSVRC2015/Data/VID/train',r'train_seq.npy') 
get_path('val',r'ILSVRC2015/Data/VID/val',r'val_seq.npy')
get_path('val',r'ILSVRC2015/Data/VID/val_augmented',r'val_seq.npy')

floder = KFold(n_splits=5, random_state=100, shuffle=True)
train_files = []   # 存放5折的训练集划分
test_files = []     # # 存放5折的测试集集划分
# 5 fold 
for k, (Trindex, Tsindex) in enumerate(floder.split(all_files)):
    # here you can select which fold to train/test
    if k == 3:
        np.save(r'train_seq.npy', np.array(all_files)[Trindex], allow_pickle=True, fix_imports=True)
        np.save(r'val_seq.npy', np.array(all_files)[Tsindex], allow_pickle=True, fix_imports=True)
#         train_files.append(np.array(all_files)[Trindex])
#         test_files.append(np.array(all_files)[Tsindex])
# np.save(r'train_seq.npy', train_files, allow_pickle=True, fix_imports=True)
# np.save(r'val_seq.npy', test_files, allow_pickle=True, fix_imports=True)