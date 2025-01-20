import os
import glob as gb
import torch

# Clear GPU memory
torch.cuda.empty_cache()

data_path = '../framesList'
rgb_path = data_path + '/JPEGImages/480p'  
# '/JPEGImages/480p' for DAVIS-related datasets and '/JPEGImages' for others

gap = [1,2]
reverse = [0, 1]
batch_size = 2  # Reduce batch size

folder = gb.glob(os.path.join(rgb_path, '*'))
for r in reverse:
    for g in gap:
        for f in folder:
            print('===> Running {}, gap {}'.format(f, g))
            mode = 'raft-things.pth'  # model
            if r == 1:
                raw_outroot = data_path + '/Flows_gap-{}/'.format(g)  # where to raw flow
                outroot = data_path + '/FlowImages_gap-{}/'.format(g)  # where to save the image flow
            elif r == 0:
                raw_outroot = data_path + '/Flows_gap{}/'.format(g)   # where to raw flow
                outroot = data_path + '/FlowImages_gap{}/'.format(g)   # where to save the image flow
                
            os.system("python predict.py "
                        "--gap {} --mode {} --path {} --batch_size {} "
                        "--outroot {} --reverse {} --raw_outroot {}".format(g, mode, f, batch_size, outroot, r, raw_outroot))