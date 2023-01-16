import torch
import torch.nn as nn

# device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

inputs = torch.Tensor(1, 1, 28, 28)
print('텐서의 크기 : {}'.format(inputs.shape))








if __name__ == '__main__':
    print('Welcomeback Umin')