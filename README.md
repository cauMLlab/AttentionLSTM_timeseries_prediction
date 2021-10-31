# Deep learning for stock price prediction
![주식_이미지](https://user-images.githubusercontent.com/76574427/139482743-d017ba50-dacd-4642-8560-29f2af7169b4.jpg)

주가 예측은 힘들지만 금융산업에서 중요한 문제임.

최근 딥러닝을 이용한 주가예측 알고리즘이 뛰어난 성능을 보이고 있음.  

아래에는 딥러닝을 이용한 방식 중 AttentionLSTM, TransformerEncoder를 이용한 예측방식을 소개함.

## Installation
- Install pytorch 
- Install numpy by running ```pip install numpy```.
- Install os by running ```pip install os```.
- Install csv by running ```pip install csv```.
- Install matplotlib by running ```pip install matplotlib```.
- Install argparse by running ```pip install argparse```.
- Install pandas by running ```pip install pandas```.
- Install pandas_datareader by running ```pip install pandas-datareader```.
- Install yfinance by running ```pip install yfinance```.
- Install math by running ```pip install python-math```.
- Install sklearn by running ```pip install scikit-learn```.


## Running
run ```AttentionLSTM_main.py``` and ```transformer_main.py```.


## Output
종목별 예측 결과, Time split별 결과와 그래프

## AttentionLSTM

본 방식은 EncoderDecoderLSTM에 Attrntion구조를 추가한 방식으로

일반적인 LSTM의 Gradient Vanishing 문제를 보완하기 위해 개발 됨.

- AttentionLSTM for one step ahead prediction

![attLSTM](https://user-images.githubusercontent.com/76574427/139543299-e7b72728-6cc6-407b-899c-261314d958aa.PNG)



## TransformerEncoder
자연어 처리에 뛰어난 성과를 보인 Transformer 구조에서
Encoder 구조만을 부분적으로 차용, Time series 데이터에 이용함.

- TransformerEncoder

![AttentionEncoder](https://user-images.githubusercontent.com/76574427/139543290-4f952916-39b6-411e-9ba1-f228b74b450d.PNG)



## Experiment setting
1. Data split

![nasted_cv](https://user-images.githubusercontent.com/76574427/139542833-d78683f0-293b-4549-8b3a-c67d19e77f3e.PNG)


데이터 분할방식으로는 Nasted Time series Cross validation을 사용.

2. metric
- MAE
- RMSE
- MAPE

3. Deeplearning library
- pytorch


## Datasets
```
from pandas_datareader import data as pdr
import yfinance as yfin

yfin.pdr_override()
self.data = pdr.get_data_yahoo(self.symbol, start=self.start, end=self.end)
```
pandas_datareader를 이용하여 야후 파이낸스에 있는 데이터셋을 위와같은 방법으로 불러올 수 있음.

yahoofinance에서 제공되는 정보(Open, Close, High, Low, Volume, AdjClose)를 불러옴.

## 예측 결과 예시

코스닥 주가 데이터를 이용.

![예측 결과 임지](https://user-images.githubusercontent.com/76574427/139482798-87decde6-a9b9-458d-9e58-f43469498780.png)
