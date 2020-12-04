---
layout: post  
title:  "[인공지능개론] 12. Recurrent Nerual Network"  
subtitle:   "수업 내용 정리하기"  
categories: study  
tags: AI RNN  
comments: false
--- 

> 인공지능을 차근차근 살펴봅시다.  

이 글은 [KAIST 홍승훈 교수님](https://maga33.github.io/)의 인공지능개론 수업을 듣고 요약 정리했음을 밝힙니다.

## 목차

---  

$$
\lim_{x\to 0}{\frac{e^x-1}{2x}}
\overset{\left[\frac{0}{0}\right]}{\underset{\mathrm{H}}{=}}
\lim_{x\to 0}{\frac{e^x}{2}}={\frac{1}{2}}
$$  

1. [RNN이란?](#RNN이란?)  
2. [RNN의 등장](#RNN의-등장)  
3. [RNN의 흐름](#RNN의-흐름)
4. [RNN의 역전파](#RNN의-역전파)
5. [vanishing gradient problem](#vanishing-gradient-problem)
6. [개선 방안](#개선-방안)

<br>

## RNN이란?

---

RNN이란 Recurrent Nerual Network의 줄임말로, <span style="padding: 0 5px; background: linear-gradient(transparent 65%, #ffb2b7 66%, #ffb2b7 100%);">sequential data</span>를 처리하기에 적합한 모델입니다. CNN의 경우 주로 사진 등 discrete data의 분류를 위해 사용했는데, RNN은 음성, 텍스트 등 **데이터에 순서**가 있고, 뒤쪽 데이터가 앞쪽 데이터에 영향을 받는 데이터를 처리합니다.

<br>

## RNN의 등장  

---  

Sequential data의 다양한 처리 방법에 대해 살펴봅시다.

* <p style="font-size: 1.05em; font-weight: bold; margin-top: 32px">1. MLP (Multi Layer Perceptron)</p>  
  먼저 가장 간단한 예시로 길이가 4로 고정된 sequence가 들어온 경우를 먼저 생각해봅시다.  
  ![MLP 예시](https://jjungkang2.github.io/assets/img/2020-11-24-AI-1.PNG)  
  사진에서 볼 수 있듯이 input x가 d, hidden layer h가 m, output y가 n차원이라고 두면 input에서 hidden layer로 향할 때 m\*4d, hidden layer에서 output으로 향할 때 4n\*m개의 파라미터로 sequential data를 처리할 수 있습니다. 그러나 sequence의 길이에 따라 파라미터의 개수가 바뀌어야하기 때문에 길이가 가변적인 dataset에서는 사용하기에 적합하지 않습니다.  
  
* <p style="font-size: 1.05em; font-weight: bold; margin-top: 32px">2. CNN (Convolutional Neural Network)</p>  
  다음으로는 CNN을 사용하는 경우를 생각해봅시다.  
  ![CNN 예시](https://jjungkang2.github.io/assets/img/2020-11-24-AI-2.PNG)  
  CNN은 fixed size 문제를 해결하기 위해 sliding convolution filter을 사용할 수 있어, MLP에서는 해결하지 못했던 길이가 달라지는 문제를 커버할 수 있습니다. 그러나 receptive field가 고정돼있어 hidden representation이 계속 늘어남에 따라 파라미터 수가 계속 늘어나게 됩니다. 따라서 CNN도 길이가 긴 sequential data를 처리하기에 최적이라고 볼 순 없습니다.  
  
* <p style="font-size: 1.05em; font-weight: bold; margin-top: 32px">3. RNN (Recurrent Neural Network)</p>  
  이러한 배경 속에 RNN이 등장하게 되었습니다.
  ![RNN 예시](https://jjungkang2.github.io/assets/img/2020-11-24-AI-3.PNG)  
  이름에 Recurrent가 들어가죠, 반복되며 사용된다는 뜻입니다. input을 한번에 하나씩만 넣고, 파라미터를 재사용하거나 공유합니다. 따라서 CNN과 파라미터가 계속 늘어나지 않는다는 장점이 있습니다!

<br>

## RNN의 흐름  

---

RNN의 학습 방법을 시간의 흐름에 따라 펼쳐서 살펴봅시다.  
![RNN 흐름](https://jjungkang2.github.io/assets/img/2020-11-24-AI-4.PNG)  


  


이전 값들을 U h_t-1로 받아서 모든 이전 값에 영향을 받음

BPTT (Back Propagation Through Time)의 문제점은 L_T 미분 h_t가 vanish, 혹은 explode할 수 있음.
gradient of sigmoid가 1보다 작으면 계속 곱해지는 과정에서 0이 되고,
U가 엄청 커서 grad랑 곱한게 1보다 크면 계속 곱해지는 과정에서 발산함.
딱 1이여야함
-> U를 orthonormal하게 함.
orthonormal = orthogonal + (norm=1)
U^T U = I
-> gradients를 clip함.
gradient가 갑자기 너무 크게 움직이면 학습했던 것들이 무효화 될 수 있기 때문에, learning rate를 작게하거나, gradient의 최대 갯수를 제한하거나, 너무 많이 변할 시 재조정 하는 등의 과정을 거침.

Option 4. LSTM (Long Short-Term Memory)

RNN은 관련 정보와 그 정보를 사용하는 지점 사이 거리가 멀 경우 역전파시 그래디언트가 점차 줄어 듬 -> vanishing gradient problem

cell state를 추가해 vanishing/exploding gradients를 막음.
중요한 아이디어는 previous memories를 "더해서" grad가 1이 되도록 함.
switch variable은 4개임

switch 1. forgate gate
'과거 정보를 잊기'
previous memories를 고려할지 안할지 결정

switch 2. input gate
'현재 정보를 기억하기'
current input을 고려할지 안할지 결정
output은 current input이랑 previous hidden unit에 의해 결정함.
그 뒤에 쓰기로 한 것을 더함
활성 함수는 tanh임

switch 3. output gate
current memories 중 다음 hidden state로 갈 것 계산함
활성 함수는 tanh임

Option 5. GRU (Gated Recurrent Unit)
LSTM에서 변수를 줄임.
gorget, input, output gate가 하나의 gate로 줄여짐.
cell이 없어짐.