from pandas_datareader import data as pdr
import yfinance as yfin

from torch.utils.data import Dataset
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit




## t+1 부터 t+5 까지 동시에 예측
## data preperation

## dataset은 i번째 record 값을 출력해주는 역할을 함.
## if input feature is 2 demension, then 인풋으로는 2차원을 주고, 아웃풋으로는 1차원 값 하나를 return 하면 끝나는 역할

## batch를 만들어주는것이 pytorch의 data loader임, random sample도 만들어줌
## data loader는 파이토치에 구현되어있음

class Data_Spliter_CrossVal:
    def __init__(self, symbol, data_start, data_end,n_splits,gap=0):
        self.symbol = symbol
        self.n_splits = n_splits

        self.start = data_start
        self.end = data_end
        yfin.pdr_override()
        self.data = pdr.get_data_yahoo(self.symbol, start=self.start, end=self.end)

        self.chart_data = self.data
        self.test_size = len(self.data)//10-1
        self.gap = gap
        print(self.data.isna().sum())

        self.tscv = TimeSeriesSplit(gap=self.gap, max_train_size=None, n_splits=self.n_splits, test_size=self.test_size)

    def ts_cv_List(self):
        list = []
        for train_index, test_index in self.tscv.split(self.data):
            X_train, X_test = self.data.iloc[train_index, :], self.data.iloc[test_index, :]
            list.append((X_train, X_test))
        return list

    def test_size(self):
        return self.test_size

    def entire_data(self):
        return self.chart_data

    def __len__(self):
        return self.n_splits

    def __getitem__(self, item):
        datalist = self.ts_cv_List()
        return datalist[item]


class Train_Data_Spliter_CrossVal:
    def __init__(self, data, symbol, test_size, gap=0):
        self.symbol = symbol
        self.data = data
        self.test_size = test_size
        self.gap = gap
        print(self.data.isna().sum())
        self.tscv = TimeSeriesSplit(gap=self.gap, max_train_size=None, n_splits=2, test_size=self.test_size)

    def ts_cv_List(self):
        list= []
        for train_index, test_index in self.tscv.split(self.data):
            X_train, X_test = self.data.iloc[train_index, :], self.data.iloc[test_index, :]
            list.append((X_train, X_test))
        return list

    def __getitem__(self, item):
        datalist= self.ts_cv_List()
        return datalist[item]


class StockDataset(Dataset):

    def __init__(self, data, x_frames, y_frames):
        self.x_frames = x_frames
        self.y_frames = y_frames
        self.data = data
        print(self.data.isna().sum())

    ## 데이터셋에 len() 을 사용하기 위해 만들어주는것 (dataloader에서 batch를 만들때 이용됨)
    def __len__(self):
        return len(self.data) - (self.x_frames + self.y_frames) + 1

    ## a[:]와 같은 indexing 을 위해 getinem 을 만듬
    ## custom dataset이 list가 아님에도 그 데이터셋의 i번째의 x,y를 출력해줌
    def __getitem__(self, idx):
        idx += self.x_frames
        data = pd.DataFrame(self.data).iloc[idx - self.x_frames:idx + self.y_frames]
        #data = data[['High', 'Low', 'Open', 'Close', 'Adj Close', 'Volume']] ## 컬럼순서맞추기 위해 한것
        data = data['Close']
        ## log nomalization
        # data = data.apply(lambda x: np.log(x + 1) - np.log(x[self.x_frames - 1] + 1))
        ## min max normalization
        min_data, max_data = min(data), max(data)
        normed_data = (data-min(data))/(max(data)-min(data))
        data = normed_data.values ## (data.frame >> numpy array) convert >> 나중에 dataloader가 취합해줌
        ## x와 y기준으로 split
        X = data[:self.x_frames]
        y = data[self.x_frames:]

        return X, y, min_data, max_data

