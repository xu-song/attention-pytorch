# -*- coding: utf-8 -*-


######################################################################
# argument parser
import os
from argparse import ArgumentParser

def makedirs(name):
    """helper function for python 2 and 3 to call os.makedirs()
       avoiding an error if the directory to be created already exists"""

    import os, errno

    try:
        os.makedirs(name)
    except OSError as ex:
        if ex.errno == errno.EEXIST and os.path.isdir(name):
            # ignore existing directory
            pass
        else:
            # a different error happened
            raise




def get_args():
    parser = ArgumentParser(description='PyTorch/torchtext SNLI example')
    
    # Data Preparation
    parser.add_argument('--task', type=str, default='translation') # 
    parser.add_argument('--data_path', type=str, default='.data')
    parser.add_argument('--dataset', type=str, default='eng-fra') # 'translation'   # 'lang' 'xiaohuangji' 'aspect'
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--d_embed', type=int, default=100)
    parser.add_argument('--d_proj', type=int, default=300)
    parser.add_argument('--d_hidden', type=int, default=300)
    parser.add_argument('--n_layers', type=int, default=1)
    parser.add_argument('--log_every', type=int, default=50)
    
    # training parameter
    # max_length: 20,
    # use_cuda
    # encoder decoder attentin

    
    
    # optimizer:
    # criterion: NLL, # nn.CrossEntropyLoss()  lr=0.001, momentum=0.9 n_iter: 7000,
    parser.add_argument('--lr', type=float, default=.001)
    
    # log  print_every: 100
    parser.add_argument('--save_path', type=str, default='.results')
    parser.add_argument('--dev_every', type=int, default=1000)
    parser.add_argument('--n_iters', type=int, default=750)
    parser.add_argument('--print_every', type=int, default=5)
    parser.add_argument('--plot_every', type=int, default=5)
    parser.add_argument('--save_every', type=int, default=5)

    
    
    
    parser.add_argument('--dp_ratio', type=int, default=0.2)
    parser.add_argument('--no-bidirectional', action='store_false', dest='birnn')
    parser.add_argument('--preserve-case', action='store_false', dest='lower')
    parser.add_argument('--no-projection', action='store_false', dest='projection')
    parser.add_argument('--train_embed', action='store_false', dest='fix_emb')
    parser.add_argument('--gpu', type=int, default=0)
    
    parser.add_argument('--vector_cache', type=str, default=os.path.join(os.getcwd(), '.vector_cache/input_vectors.pt'))
    parser.add_argument('--word_vectors', type=str, default='glove.6B.100d') # use_pretrain
    parser.add_argument('--resume_snapshot', type=str, default='')
    args = parser.parse_args()
    return args