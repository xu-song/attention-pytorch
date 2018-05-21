
# config可以分散在不同文件。比如数据相关的，放在dataset目录的文件中。train相关的，放在train.py中


run_mode: train # or test

is_train = True
if len(sys.argv) > 1:  
    if sys.argv[1]=='eval':
        is_train = False

use_cuda = torch.cuda.is_available()
print("use_cuda:" , use_cuda)

config = 

'aspect': {
# Data Preparation
# ==================================================
name: 'SemEval 2014 Task 4',
download: '',
dataset: 'aspect',
path: '',
max_length: 20,
count_sos:  # 这个配置是和


# pretrain-model


# model

encoder: '',
decoder: '',
attentin: '',


# optimizer:
criterion: NLL, # nn.CrossEntropyLoss()
lr=0.001, momentum=0.9
n_iter: 7000,



# log

print_every: 100,
}

'seq2seq':{

        }
}
