import torch
from collections import OrderedDict
import os
import torch.nn as nn
import torch.nn.init as init

#
#加载pretrain model_dir
state_dict = torch.load(r'C:\Users\Tom.riddle\Desktop\代码\ghfeat-pytorch\model\Generator.pth')


keys = []
original_keys=[]
for k,v in state_dict.items():
    print(k)
    if k.startswith('g_synthesis'):    #将‘conv_cls’开头的key过滤掉，这里是要去除的层的key
        original_keys.append(k)
        keys.append(k[12:])

for k in keys:
    print(k)

new_dict = {k:state_dict[k_o] for k_o,k in zip(original_keys,keys)}
print(new_dict)
torch.save(new_dict, r'C:\Users\Tom.riddle\Desktop\代码\ghfeat-pytorch\model/gsynthesis.pth')
# new_dict = {k:state_dict[k] for k in keys}
# # stylor.load_state_dict(new_dict)
# torch.save(new_dict, r'C:\Users\Tom.riddle\Desktop\代码\representation GAN3\model_dir/stylor.pth')
#
#
#
# state_dict = torch.load(r'C:\Users\Tom.riddle\Desktop\代码\ghfeat-pytorch\model\Generator.pth')
# keys = []
# original_keys = []
# for k,v in state_dict.items():
#     print(k)
#     original_keys.append(k)
#     keys.append(k[2:])
#
# print(keys)
# new_dict = {k:state_dict[k_o] for k_o,k in zip(original_keys,keys)}
# torch.save(new_dict, r'C:\Users\Tom.riddle\Desktop\代码\stylegan2-pytorch-master\stylegan2\models\100\gen.pth')

