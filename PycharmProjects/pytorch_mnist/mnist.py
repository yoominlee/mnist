'''
# 1번 레이어 : 합성곱층(Convolutional layer)
합성곱(in_channel = 1, out_channel = 32, kernel_size=3, stride=1, padding=1) + 활성화 함수 ReLU
맥스풀링(kernel_size=2, stride=2))

# 2번 레이어 : 합성곱층(Convolutional layer)
합성곱(in_channel = 32, out_channel = 64, kernel_size=3, stride=1, padding=1) + 활성화 함수 ReLU
맥스풀링(kernel_size=2, stride=2))

# 3번 레이어 : 전결합층(Fully-Connected layer)
특성맵을 펼친다. # batch_size × 7 × 7 × 64 → batch_size × 3136
전결합층(뉴런 10개) + 활성화 함수 Softmax
'''





# # --- 1. 필요한 도구 임포트와 입력의 정의 ---
# import torch
# import torch.nn as nn
#
# # 임의의 텐서를 만듭니다. 텐서의 크기는 1 × 1 × 28 × 28
# # 배치 크기 × 채널 × 높이(height) × 너비(widht)의 크기의 텐서를 선언
# inputs = torch.Tensor(1, 1, 28, 28)
# print('텐서의 크기 : {}'.format(inputs.shape)) # 텐서의 크기 : torch.Size([1, 1, 28, 28])
# # print(inputs)
# print('>>>>>>>>>>><<<<<<<<<<<<<<<<<<')
#
# # --- 2. 합성곱층과 풀링 선언하기 ---
# # 첫번째 합성곱 층 구현. 1채널 짜리를 입력받아서 32채널을 뽑아내는데 커널 사이즈 3이고 패딩 1입니다.
#
# conv1 = nn.Conv2d(1, 32, 3, padding=1)
# print(conv1) # Conv2d(1, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
# print(conv1.weight)
# print(conv1.weight.shape) # torch.Size([32, 1, 3, 3]) ---> weight가 가지는 shape : batch_size, channels, height, width의 크기
#
# # show weight
# weight = conv1.weight.detach().numpy()
#
# # plt.imshow(weight[0, 0, :, :], 'jet')
# # plt.colorbar()
# # plt.show()
#
# # 두번째 합성곱 층 구현. 32채널 짜리를 입력받아서 64채널을 뽑아내는데, 커널 사이즈 3 패딩 1.
# conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
# print(conv2) # Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
# print(conv2.weight.shape) # torch.Size([32, 1, 3, 3]) ---> weight가 가지는 shape : batch_size, channels, height, width의 크기
#
# # weight2 = conv2.weight.detach().numpy()
# # plt.imshow(weight2[0, 0, :, :], 'jet')
# # plt.colorbar()
# # plt.show()
#
# # 맥스풀링 구현. 정수 하나를 인자로 넣으면 커널 사이즈와 스트라이드가 둘 다 해당값으로 지정.
# pool = nn.MaxPool2d(2)
# print(pool) # MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
#
# # --- 3. 구현체를 연결하여 모델 만들기 ---
# # 지금까지는 선언만
# # 이들을 연결시켜서 모델을 완성
#
# # 3-1. 입력을 첫번째 합성곱층을 통과시키고 합성곱층을 통과시킨 후의 텐서의 크기 출력
# out = conv1(inputs)
# print(out.shape) # torch.Size([1, 32, 28, 28]) --> 32채널의 28너비 28높이의 텐서
#
# '''
# 32가 나온 이유는 conv1의 out_channel로 32를 지정해주었기 때문.
# 28너비 28높이가 된 이유는 패딩을 1폭으로 하고 3 × 3 커널을 사용하면 크기가 보존되기 때문.
# '''
#
# # 3-2. 이제 이를 맥스풀링을 통과시키고 맥스풀링을 통과한 후의 텐서의 크기 확인
#
# out = pool(out)
# print(out.shape) # torch.Size([1, 32, 14, 14]) ---> 32채널의 14너비 14높이의 텐서가 됨
#
# # 두번째 합성곱층에 통과시키고 통과한 후의 텐서의 크기 확인
# out = conv2(out)
# print(out.shape) # torch.Size([1, 64, 14, 14]) ---> 64채널의 14너비 14높이의 텐서가 됨
# '''
# 64가 나온 이유는 conv2의 out_channel로 64를 지정해주었기 때문
# 14너비 14높이가 된 이유는 패딩을 1폭으로 하고 3 × 3 커널을 사용하면 크기가 보존되기 때문
# '''
#
# # 3-3. 맥스풀링을 통과시키고 맥스풀링을 통과한 후의 텐서의 크기 확인
#
# out = pool(out)
# print(out.shape) # torch.Size([1, 64, 7, 7])
#
# # 3-4. 텐서를 펼치는 작업
#
# '''
# 텐서의 n번째 차원을 접근하게 해주는 .size(n)
# 현재 out의 크기는 1 × 64 × 7 × 7
# '''
# # out의 첫번째 차원이 몇인지 출력
# print('{}번째 out.size={}'.format(1,out.size(0))) # 1
# # out의 두번째 차원이 몇인지 출력
# print('{}번째 out.size={}'.format(2,out.size(1))) # 64
# # out의 세번째 차원이 몇인지 출력
# print('{}번째 out.size={}'.format(3,out.size(2))) # 7
# # out의 네번째 차원이 몇인지 출력
# print('{}번째 out.size={}'.format(4,out.size(3))) # 7
#
# # 이제 이를 가지고 .view()를 사용하여 텐서를 펼치는 작업
#
# # 첫번째 차원인 배치 차원은 그대로 두고 나머지는 펼쳐라
# out = out.view(out.size(0), -1)
# print(out.shape) # torch.Size([1, 3136]) ---> 배치 차원을 제외하고 모두 하나의 차원으로 통합 됨
#
# # 이에 대해 전결합층(Fully-Connteced layer)를 통과
# # 출력층으로 10개의 뉴런을 배치하여 10개 차원의 텐서로 변환
#
# fc = nn.Linear(3136, 10) # input_dim = 3,136, output_dim = 10
# out = fc(out)
# print(out.shape) # torch.Size([1, 10])
#
# print('===================import torch========================')

import torch
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torch.nn.init

import matplotlib.pyplot as plt


'''
torch.cuda.is_available()
# cuda가 사용 가능하면 true를 반환함으로서 device에 cuda를 설정하도록 한다
# gpu 사용이 가능한지?

device에 cuda (GPU)를 설정하고,
device와 current_device()를 출력해봄으로 GPU가 잘 할당되었는지 확인가능
'''
device = 'cuda' if torch.cuda.is_available() else 'cpu'


'''
manual_seed()
# 랜덤 시드 고정 = 동일한 셋트의 난수를 생성할 수 있게 하는 것
# 반환값으로는 generator를 반환
# 괄호 안에 들어가는 숫자 자체는 중요하지 않고 서로 다른 시드를 사용하면 서로 다른 난수를 생성한다
'''
torch.manual_seed(777)

# gpu 사용 설정 후, random_value를 위한 seed 설정
# for reproducibility

# GPU 사용 가능일 경우 랜덤 시드 고정
if device == 'cuda':
    torch.cuda.manual_seed_all(777) # 동일한 조건에서 학습시 weight가 변화하지 않게 하는 옵션

# 학습에 사용할 파라미터를 설정
learning_rate = 0.001
training_epochs = 0 # ================================================= epoch
batch_size = 100

# 데이터로더를 사용하여 데이터를 다루기 위해서 데이터셋을 정의
# MNIST dataset
'''
torchvision.datasets 모듈의 MNIST객체로 MNIST 데이터를 불러와서 데이터 셋 객체를 만든다.

root : 데이터의 경로
transform : 어떤 형태로 데이터를 불러올 것인가
transform : 일반 이미지는 0-255사이의 값을 갖고, (H, W, C)의 형태를 갖는 반면 
            pytorch는 0-1사이의 값을 가지고 (C, H, W)의 형태를 갖는다. 
            transform에 transforms.ToTensor()를 넣어서 일반 이미지(PIL image)를 pytorch tensor로 변환

'''
mnist_train = dsets.MNIST(root='MNIST_data/', # 다운로드 경로 지정
                          train=True, # True를 지정하면 훈련 데이터로 다운로드
                          transform=transforms.ToTensor(), # 텐서로 변환
                          download=True)

mnist_test = dsets.MNIST(root='MNIST_data/', # 다운로드 경로 지정
                         train=False, # False를 지정하면 테스트 데이터로 다운로드
                         transform=transforms.ToTensor(), # 텐서로 변환
                         download=True)

print(mnist_test[0]) # 0-1 data
'''
위에서 불러온 데이터셋 객체로 이제 data_loader 객체를 만든다. 

dataset : 어떤 데이터를 로드할 것인지
batch_size : 배치 사이즈를 뭘로 할지
shuffle : 순서를 무작위로 할 것인지, 있는 순서대로 할 것인지
drop_last : batch_size로 자를 때 맨 마지막에 남는 데이터를 사용할 것인가 버릴 것인가

'''
# 데이터로더를 사용하여 배치 크기를 지정 batch_size와 dataset을 통해 dataLoder 로딩
# 미니 배치와 데이터 로드 보기 --> https://wikidocs.net/55580
data_loader = torch.utils.data.DataLoader(dataset=mnist_train,
                                          batch_size=batch_size,
                                          shuffle=True,
                                          drop_last=True)

# 클래스로 모델을 설계
class CNN(torch.nn.Module):                     #---------------------##

    def __init__(self):                         #---------------------##
        super(CNN, self).__init__()             #---------------------##

        # nn.Sequential은 순서를 갖는 모듈의 컨테이너. 데이터는 정의된 것과 같은 순서로 모든 모듈들을 통해 전달

        # 첫번째층
        # ImgIn shape=(?, 28, 28, 1)
        #    Conv     -> (?, 28, 28, 32)
        #    Pool     -> (?, 14, 14, 32)
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2))

        # 두번째층
        # ImgIn shape=(?, 14, 14, 32)
        #    Conv      ->(?, 14, 14, 64)
        #    Pool      ->(?, 7, 7, 64)
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2))


        '''
        torch.nn.Linear(in_features,out_features,bias = True, device = None,dtype = None)

        bias는 만일 False로 설정되어 있으면 layer는 bias를 학습하지 않는다
        '''
        # Final FC (전결합층) 7x7x64 inputs -> 10 outputs
        self.fc = torch.nn.Linear(7 * 7 * 64, 10, bias=True)

        # 전결합층 한정으로 가중치 초기화
        torch.nn.init.xavier_uniform_(self.fc.weight)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)   # 전결합층(FC)을 위해서 Flatten
        out = self.fc(out)
        return out

# 모델을 정의
# CNN 모델 정의

# GPU에 ``CNN()``의 복사본이 반환
model = CNN().to(device)

# 비용 함수와 옵티마이저를 정의
criterion = torch.nn.CrossEntropyLoss().to(device)    # 비용 함수에 소프트맥스 함수 포함되어져 있음.
'''
Adam은 가장 흔하게 이용되는 옵티마이저로 Momentum에 적용된 그래디언트 조정법과 Adagrad에 적용된 학습률 조정법의 장점을 융합한 옵티마이저
'''
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# 총 배치의 수를 출력
total_batch = len(data_loader)
print('총 배치의 수 : {}'.format(total_batch)) # 총 배치의 수 : 600
                                            # 배치 크기를 100으로 했으므로 결국 훈련 데이터는 총 60,000개

'''
학습을 돌린다.
for문이 한번 돌때마다 batch_size만큼의 데이터를 꺼내서 학습시키는 것
for문이 한번 돌면 1 epoch만큼 학습시킨 것

'''
# 모델을 훈련
for epoch in range(training_epochs):
    avg_cost = 0
    count = 0
    for X, Y in data_loader: # mnist_train 미니 배치 단위로 꺼내온다. X는 미니 배치(=image), Y는 레이블.
        # image is already size of (28x28), no reshape
        # label is not one-hot encoded
        X = X.to(device)
        Y = Y.to(device)

        optimizer.zero_grad()   # 파이토치에는 미분값 누적하는 특징. 따라서 미분을 통해 얻은 기울기를 0으로 초기화
        hypothesis = model(X)  # hypothesis = 모델이자 가설
        cost = criterion(hypothesis, Y)       # loss = criterion(outputs, labels)
        cost.backward()
        optimizer.step()


        # print('-------------------------- {} ----------------------'.format(count))
        # print('cost: {}'.format(cost))
        # print('total_batch: {}'.format(total_batch))
        # print('avg_cost: {}'.format(avg_cost))

        avg_cost += cost / total_batch
        count = count + 1

    print('[Epoch: {:>4}] cost = {:>.9}'.format(epoch + 1, avg_cost))

    '''
    [Epoch:    1] cost = 0.223893151
    [Epoch:    2] cost = 0.0621390603
    [Epoch:    3] cost = 0.0450111665
    [Epoch:    4] cost = 0.0355613008
    [Epoch:    5] cost = 0.0290638003
    [Epoch:    6] cost = 0.0249994267
    [Epoch:    7] cost = 0.0206681583
    [Epoch:    8] cost = 0.0180210788
    [Epoch:    9] cost = 0.015301072
    [Epoch:   10] cost = 0.0126666417
    [Epoch:   11] cost = 0.0107044494
    [Epoch:   12] cost = 0.0101312753
    [Epoch:   13] cost = 0.00786222517
    [Epoch:   14] cost = 0.00772933941
    [Epoch:   15] cost = 0.00640067318
    '''

print('===================테스트========================')
# torch.save(model, '/home/cona/mnist/PycharmProjects/pytorch_mnist/model_result/mnist.pth')
import random

model_load = torch.load('/home/cona/mnist/PycharmProjects/pytorch_mnist/model_result/mnist.pth')
# 테스트
# 학습을 진행하지 않을 것이므로 torch.no_grad() = 이 범위 안에서느 gradient 계산을 안하겠다는 의미
with torch.no_grad():
    '''
    뷰(View) - 원소의 수를 유지하면서 텐서의 크기 변경
    '''
    X_test = mnist_test.test_data.view(len(mnist_test), 1, 28, 28).float().to(device)
    Y_test = mnist_test.test_labels.to(device)

    print('mnist_test.test_data.size() : ', mnist_test.test_data.size()) # torch.Size([10000, 28, 28])
    # print(X_test.type()) # torch.cuda.FloatTensor

    prediction = model_load(X_test)
    '''
    torch.argmax = input tensor에 있는 모든 element들 중에서 가장 큰 값을 가지는 공간의 인덱스 번호를 반환하는 함수
    '''
    # 예측된 결과와 실제 test label 간의 맞는 정도
    correct_prediction = torch.argmax(prediction, 1) == Y_test
    accuracy = correct_prediction.float().mean()
    print('Accuracy:', accuracy.item()) # Accuracy: 0.9883000254631042

    # print('Prediction.shape:', prediction.shape)  # Prediction.shape: torch.Size([10000, 10])
    # print('Prediction:', prediction)  # Prediction: torch.Size([10000, 10])







    '''
    랜덤으로 이미지를 하나 골라서 테스트하고, 그 이미지를 pyplot으로 시각화
    
    '''
    # # Get one and predict
    # r = random.randint(0, len(mnist_test) - 1)
    # X_single_data = mnist_test.test_data[r:r + 1].view(-1, 28 * 28).float().to(device)
    # Y_single_data = mnist_test.test_labels[r:r + 1].to(device)
    #
    # print('Label: ', Y_single_data.item())
    # single_prediction = model(X_single_data)
    # print('Prediction: ', torch.argmax(single_prediction, 1).item())
    #
    # plt.imshow(mnist_test.test_data[r:r + 1].view(28, 28), cmap='Greys', interpolation='nearest')
    # plt.show()

    # r = random.randint(0, len(mnist_test) - 1)
    # X_single_data = mnist_test.test_data[r:r + 1].view(-1, 28 * 28).float().to(device)
    # Y_single_data = mnist_test.test_labels[r:r + 1].to(device)
    #
    # print('Label: ', Y_single_data.item())
    # single_prediction = linear(X_single_data)
    # print('Prediction: ', torch.argmax(single_prediction, 1).item())
    #
    # plt.imshow(mnist_test.test_data[r:r + 1].view(28, 28), cmap='Greys', interpolation='nearest')
    # plt.show()






















'''
이미지 테스트 #################################################################################################
'''
import PIL

# a-start. 저장된 이미지 imshow --------------------------------------------------------------
# img = PIL.Image.open('/home/cona/mnist/PycharmProjects/pytorch_mnist/MNIST_data/my_data/0/0.jpg')
# tf = transforms.ToTensor()
# img_t = tf(img)
# print(img_t.size())
# img_t = img_t.permute(1,2,0)    # permute : tensor의 모양을 바꾸는데 사용
# print(img_t.size())
#
# plt.imshow(img_t)
# plt.show()
# a-fin -----------------------------------------------------------------------------------

# b-start. 내 이미지 dataset으로 --------------------------------------------------------------
import torchvision
from torchvision import transforms
trans = transforms.Compose([transforms.Grayscale(),
                            transforms.Resize((28,28)),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)) # Norm X -> Accuracy: 0.20000000298023224 / Norm O -> Accuracy: 0.25
                                                                            # -1 ~ 1 사이의 범위를 가지도록 정규화
                            ])

from torchvision.datasets import ImageFolder
myset_imgfolder = ImageFolder(root = "/home/cona/mnist/PycharmProjects/pytorch_mnist/MNIST_data/my_data/test",
                              transform = trans)
print('len(myset) : ',len(myset_imgfolder)) # len(myset) :  20
classes = myset_imgfolder.classes
# print('myset.classes : ',classes) # myset.classes :  ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']


myset_loder = torch.utils.data.DataLoader(dataset=myset_imgfolder,batch_size=20,
                                          shuffle=False, num_workers=2)

# print('myset_imgfolder[0][0].size() : ',myset_imgfolder[0][0].size()) # myset_imgfolder[0][0].size() :  torch.Size([1, 28, 28])

data_iter = iter(myset_loder)
images, labels = data_iter.next()

import numpy as np
def imageshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()
imageshow(torchvision.utils.make_grid(images))        ######## 여러이미지 imshow !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


print('labels : ',labels) # shuffle X -> labels :  tensor([0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9])
print('>>>',labels[0].item())
print('>>>',labels[1].item())
print('>>>',labels[2].item())
print('>>>',labels[3].item())
print('images[0].shape : ',images[0].shape)
print('labels[0].shape : ',labels[0].shape)

# 맨 앞에 이미지 normalize된거 imshow
# plt.imshow(np.transpose(images[0], (1, 2, 0)))
# print(np_img.shape)
# print((np.transpose(np_img, (1, 2, 0))).shape)
# plt.show()

print('len(myset_loder) : ',len(myset_loder))

print('len(myset_imgfolder) : ',len(myset_imgfolder))

# print(images)
# # Let's see what if the model identifiers the  labels of those example
model_load2 = torch.load('/home/cona/mnist/PycharmProjects/pytorch_mnist/model_result/mnist.pth')
images = images.to("cuda:0")
labels = labels.to("cuda:0")

outputs = model_load2(images)
# print(outputs)


# 예측된 결과와 실제 test label 간의 맞는 정도
correct_prediction = torch.argmax(outputs, 1) == labels
# print(torch.argmax(outputs, 1) == labels)       # tensor([ True, False, False,  True, False, False, False, False, False, False, True,  True, False, False, False, False,  True, False, False, False], device='cuda:0')
my_accuracy = correct_prediction.float().mean()
# print(correct_prediction.float())               # tensor([1., 0., 0., 1., 0., 0., 0., 0., 0., 0., 1., 1., 0., 0., 0., 0., 1., 0., 0., 0.], device='cuda:0')
print('My Accuracy:', my_accuracy.item()) # Accuracy: 0.25

# print(torch.argmax(outputs, 1)) # tensor([0, 6, 8, 1, 0, 1, 8, 8, 8, 5, 5, 5, 3, 3, 8, 8, 8, 5, 8, 5], device='cuda:0')
# print(labels) # tensor([0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9], device='cuda:0')

# for a in range(20):
#     print('[labels: {:>4}] correct_prediction = {:>.9}'.format(labels[a], correct_prediction[a]))




#
import glob

x_temp = torch.argmax(outputs, 1).tolist()
print(x_temp)

for index, value in enumerate(x_temp):
    x_temp[index] = str(value)

x_temp.insert(0, "xlabel")
print(x_temp)

import cv2


fig = plt.figure()  # rows*cols 행렬의 i번째 subplot 생성
rows = 4
cols = 5
i = 1
images = images.to("cpu")
from PIL import Image
print(images.size()) # torch.Size([20, 1, 28, 28])
# np_arr = np.array(images, dtype=np.uint8)
# images = PIL.Image.fromarray(np_arr)
#
for img in images[:, 0, :, :]:
    ax = fig.add_subplot(rows, cols, i)
    ax.imshow(img)
    ax.set_xlabel(x_temp[i])
    ax.set_xticks([]), ax.set_yticks([])
    i += 1

plt.show()



'''
        # ImageFolder의 속성 값인 class_to_idx를 할당
        labels_map = {v:k for k, v in myset.class_to_idx.items()}
        
        figure = plt.figure(figsize=(12, 8))
        cols, rows = 5, 4
        
        # 이미지를 출력합니다. RGB 이미지로 구성되어 있습니다.
        for i in range(1, cols * rows + 1):
            sample_idx = torch.randint(len(images), size=(1,)).item()
            img, label = images[sample_idx], labels[sample_idx].item()
            figure.add_subplot(rows, cols, i)
            plt.title(labels_map[label])
            plt.axis("off")
            # 본래 이미지의 shape은 (3, 300, 300) 입니다.
            # 이를 imshow() 함수로 이미지 시각화 하기 위하여 (300, 300, 3)으로 shape 변경을 한 후 시각화합니다.
            plt.imshow(torch.permute(img, (1, 2, 0)))
        plt.show()
'''


######################################################################################################################