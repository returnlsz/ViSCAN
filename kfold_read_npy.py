import re
import numpy as np

# mode = 'train'
mode = 'val'
fold = 4
file = np.load(mode+r'_seq.npy',allow_pickle=True)
# print(file)
np.savetxt(mode+r'_seq.txt',file,delimiter=',' ,fmt = '%s')

result = []

with open(mode+'_seq.txt') as f:
  lines = f.readlines()
  
for line in lines:
  # if line.startswith('['):
  #   continue
  
  m = re.search('ILSVRC2015_VID_train_0000/ILSVRC2015_'+'train'+'_(\d+)', line)
  if m:
    result.append('ILSVRC2015_VID_train_0000/ILSVRC2015_'+'train_'+m.group(1))
  m = re.search('ILSVRC2015_VID_train_0000_augmented/ILSVRC2015_'+'train'+'_(\d+)', line)
  if m:
    result.append('ILSVRC2015_VID_train_0000_augmented/ILSVRC2015_'+'train_'+m.group(1))
  m = re.search('NEA_DATA/ILSVRC2015_train_NEA'+'(\d+)', line)
  if m:
    result.append('NEA_DATA/ILSVRC2015_train_NEA'+m.group(1))
  m = re.search('val/ILSVRC2015_'+'val'+'_(\d+)', line)
  if m:
    result.append('val/ILSVRC2015_'+'val_'+m.group(1))
  m = re.search('val_augmented/ILSVRC2015_'+'val'+'_(\d+)', line)
  if m:
    result.append('val_augmented/ILSVRC2015_'+'val_'+m.group(1))
with open('fold/2/'+str(fold)+'fold_'+mode+'.txt', 'w') as f:
  for r in result:
    f.write(r + '\n')