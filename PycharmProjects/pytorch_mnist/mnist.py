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
# --- 1. 필요한 도구 임포트와 입력의 정의 ---
import torch
import torch.nn as nn

# 임의의 텐서를 만듭니다. 텐서의 크기는 1 × 1 × 28 × 28
# 배치 크기 × 채널 × 높이(height) × 너비(widht)의 크기의 텐서를 선언
inputs = torch.Tensor(1, 1, 28, 28)
print('텐서의 크기 : {}'.format(inputs.shape)) # 텐서의 크기 : torch.Size([1, 1, 28, 28])

# --- 2. 합성곱층과 풀링 선언하기 ---
# 첫번째 합성곱 층 구현. 1채널 짜리를 입력받아서 32채널을 뽑아내는데 커널 사이즈 3이고 패딩 1입니다.

conv1 = nn.Conv2d(1, 32, 3, padding=1)
print(conv1) # Conv2d(1, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

# 두번째 합성곱 층 구현. 32채널 짜리를 입력받아서 64채널을 뽑아내는데, 커널 사이즈 3이고 패딩 1입니다.
conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
print(conv2) # Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

# 맥스풀링 구현. 정수 하나를 인자로 넣으면 커널 사이즈와 스트라이드가 둘 다 해당값으로 지정.
pool = nn.MaxPool2d(2)
print(pool) # MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)

# --- 3. 구현체를 연결하여 모델 만들기 ---
# 지금까지는 선언만
# 이들을 연결시켜서 모델을 완성

# 3-1. 입력을 첫번째 합성곱층을 통과시키고 합성곱층을 통과시킨 후의 텐서의 크기 출력
out = conv1(inputs)
print(out.shape) # torch.Size([1, 32, 28, 28]) --> 32채널의 28너비 28높이의 텐서

'''
32가 나온 이유는 conv1의 out_channel로 32를 지정해주었기 때문. 
28너비 28높이가 된 이유는 패딩을 1폭으로 하고 3 × 3 커널을 사용하면 크기가 보존되기 때문.
'''

# 3-2. 이제 이를 맥스풀링을 통과시키고 맥스풀링을 통과한 후의 텐서의 크기 확인

out = pool(out)
print(out.shape) # torch.Size([1, 32, 14, 14]) ---> 32채널의 14너비 14높이의 텐서가 됨

# 두번째 합성곱층에 통과시키고 통과한 후의 텐서의 크기 확인
