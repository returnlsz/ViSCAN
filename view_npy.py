# Convert the .npy file to a .txt file to facilitate viewing its content.
import numpy as np
 
file = np.load(r'val_seq.npy',allow_pickle=True)
print(file)
np.savetxt(r'val_seq.txt',file,delimiter=',' ,fmt = '%s')