import sys
sys.path.append('[typrrCurrent Directory]')
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import numpy as np
import argparse
from copy import deepcopy # Add Deepcopy for args
import matplotlib.pyplot as plt
from metric import metric_mae as MAE
from metric import metric_rmse as RMSE
from metric import metric_mape as MAPE
from Stock_Dataset import StockDataset
from Stock_Dataset import Data_Spliter_CrossVal
from Stock_Dataset import Train_Data_Spliter_CrossVal
import os
import csv
from Encoder_Decoder_LSTM import LSTM_enc
from Encoder_Decoder_LSTM import LSTM_dec



def train(encoder,decoder, partition, enc_optimizer,dec_optimizer, loss_fn, args):
    trainloader = DataLoader(partition['train'],    ## DataLoader는 dataset에서 불러온 값으로 랜덤으로 배치를 만들어줌
                             batch_size=args.batch_size,
                             shuffle=False, drop_last=True)
    encoder.train()
    decoder.train()
    encoder.zero_grad()
    decoder.zero_grad()
    enc_optimizer.zero_grad()
    dec_optimizer.zero_grad()

    not_used_data_len = len(partition['train']) % args.batch_size
    train_loss = 0.0
    y_pred_graph = []
    for i, (X, y, min, max) in enumerate(trainloader):

        ## (batch size, sequence length, input dim)
        X = X.transpose(0, 1).unsqueeze(-1).float().to(args.device)

        encoder.zero_grad()
        enc_optimizer.zero_grad()
        encoder.hidden = [hidden.to(args.device) for hidden in encoder.init_hidden()]
        y_pred_enc, encoder_hidden = encoder(X)

        decoder_hidden = encoder_hidden
        y_true = y[:, :].float().to(args.device)

        decoder.zero_grad()
        dec_optimizer.zero_grad()

        y_pred_dec, atten_w = decoder(X, decoder_hidden)


        max = max.to(args.device)
        min = min.to(args.device)

        reformed_y_pred = y_pred_dec.squeeze() * (max - min) + min

        y_pred_graph = y_pred_graph + reformed_y_pred.tolist()


        loss = loss_fn(y_pred_dec.view(-1), y_true.view(-1)) # .view(-1)은 1열로 줄세운것

        loss.backward()  ## gradient 계산
        enc_optimizer.step()  ## parameter 갱신
        dec_optimizer.step()  ## parameter를 update 해줌 (.backward() 연산이 시행된다면(기울기 계산단계가 지나가면))

        train_loss += loss.item()   ## item()은 loss의 스칼라값을의미 >> cpu로 다시 넘겨줄 필요가 없음.

    train_loss = train_loss / len(trainloader)
    return encoder, decoder, train_loss, y_pred_graph,not_used_data_len


def validate(encoder, decoder, partition, loss_fn, args):
    valloader = DataLoader(partition['val'],
                           batch_size=args.batch_size,
                           shuffle=False, drop_last=True)
    not_used_data_len = len(partition['val']) % args.batch_size

    encoder.eval()
    decoder.eval()
    val_loss = 0.0

    with torch.no_grad():
        y_pred_graph = []
        for i, (X, y, min, max) in enumerate(valloader):

            X = X.transpose(0, 1).unsqueeze(-1).float().to(args.device)
            encoder.hidden = [hidden.to(args.device) for hidden in encoder.init_hidden()]
            y_pred_enc, encoder_hidden = encoder(X)

            decoder_hidden = encoder_hidden

            y_true = y[:, :].float().to(args.device)

            y_pred_dec, atten_w = decoder(X,decoder_hidden)

            max = max.to(args.device)
            min = min.to(args.device)
            reformed_y_pred = y_pred_dec.squeeze() * (max - min) + min
            y_pred_graph = y_pred_graph + reformed_y_pred.tolist()

            # print('validate y_pred: {}, y_pred.shape : {}'. format(y_pred, y_pred.shape))
            loss = loss_fn(y_pred_dec.view(-1), y_true.view(-1))

            val_loss += loss.item()


    val_loss = val_loss / len(valloader) ## 한 배치마다의 로스의 평균을 냄
    return val_loss, y_pred_graph, not_used_data_len   ## 그결과값이 한 에폭마다의 LOSS

def test(encoder,decoder, partition, args):
    testloader = DataLoader(partition['test'],
                           batch_size=args.batch_size,
                           shuffle=False, drop_last=True)
    not_used_data_len = len(partition['test']) % args.batch_size

    encoder.eval()
    decoder.eval()
    MAE_metric = 0.0
    RMSE_metric = 0.0
    MAPE_metric = 0.0
    with torch.no_grad():
        y_pred_graph = []
        for i, (X, y, min, max) in enumerate(testloader):

            X = X.transpose(0, 1).unsqueeze(-1).float().to(args.device)
            encoder.hidden = [hidden.to(args.device) for hidden in encoder.init_hidden()]
            y_pred_enc, encoder_hidden = encoder(X)

            decoder_hidden= encoder_hidden

            y_true = y[:, :].float().to(args.device)

            y_pred_dec, atten_w  = decoder(X,decoder_hidden)

            max = max.to(args.device)
            min = min.to(args.device)

            reformed_y_pred = y_pred_dec.squeeze() * (max - min) + min
            reformed_y_true = y_true.squeeze() * (max - min) + min
            y_pred_graph = y_pred_graph + reformed_y_pred.tolist()


            MAE_metric += MAE(reformed_y_pred, reformed_y_true)
            RMSE_metric += RMSE(reformed_y_pred, reformed_y_true)
            MAPE_metric += MAPE(reformed_y_pred, reformed_y_true)

    MAE_metric = MAE_metric / len(testloader)
    RMSE_metric = RMSE_metric / len(testloader)
    MAPE_metric = MAPE_metric / len(testloader)
    return MAE_metric, RMSE_metric, MAPE_metric, y_pred_graph, not_used_data_len


def experiment(partition, args):
    encoder = args.encoder(args.input_dim, args.hid_dim, args.n_layers, args.batch_size)
    decoder = args.decoder(args.input_dim, args.hid_dim, args.y_frames, args.n_layers, args.batch_size,
                           args.dropout, args.use_bn,args.attn_head,args.attn_size, args.x_frames, activation="ReLU")

    encoder.to(args.device)
    decoder.to(args.device)


    loss_fn = nn.MSELoss()
    # loss_fn.to(args.device) ## gpu로 보내줌  간혹 loss에 따라 안되는 경우도 있음
    if args.optim == 'SGD':
        enc_optimizer = optim.SGD(encoder.parameters(), lr=args.lr, weight_decay=args.l2)
        dec_optimizer = optim.SGD(decoder.parameters(), lr=args.lr, weight_decay=args.l2)
    elif args.optim == 'RMSprop':
        enc_optimizer = optim.RMSprop(encoder.parameters(), lr=args.lr, weight_decay=args.l2)
        dec_optimizer = optim.RMSprop(decoder.parameters(), lr=args.lr, weight_decay=args.l2)
    elif args.optim == 'Adam':
        enc_optimizer = optim.Adam(encoder.parameters(), lr=args.lr, weight_decay=args.l2)
        dec_optimizer = optim.Adam(decoder.parameters(), lr=args.lr, weight_decay=args.l2)
    else:
        raise ValueError('In-valid optimizer choice')

    # ===== List for epoch-wise data ====== #
    train_losses = []
    val_losses = []
    # ===================================== #

    epoch_graph_list = []## 모델마다의 디렉토리
    for epoch in range(args.epoch):  # loop over the dataset multiple times
        ts = time.time()

        encoder, decoder, train_loss, graph1, unused_triain = train(encoder, decoder, partition,
                                                                    enc_optimizer, dec_optimizer, loss_fn, args)
        val_loss, graph2, unused_val = validate(encoder, decoder, partition, loss_fn, args)
        te = time.time()

        epoch_graph_list.append([graph1,graph2])
        # ====== Add Epoch Data ====== # ## 나중에 그림그리는데 사용할것
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        # ============================ #
        ## 각 에폭마다 모델을 저장하기 위한 코드
        torch.save(encoder.state_dict(), args.innate_path + '\\' + str(epoch) +'_epoch' +'_encoder' +'.pt')
        torch.save(decoder.state_dict(), args.innate_path + '\\' + str(epoch) +'_epoch' +'_decoder' +'.pt')

        print('Epoch {}, Loss(train/val) {:2.5f}/{:2.5f}. Took {:2.2f} sec, Iteration {}'
              .format(epoch, train_loss, val_loss, te - ts, args.iteration))

    ## val_losses에서 가장 값이 최소인 위치를 저장함
    site_val_losses = val_losses.index(min(val_losses)) ## 10 epoch일 경우 0번째~9번째 까지로 나옴
    encoder = args.encoder(args.input_dim, args.hid_dim, args.n_layers, args.batch_size)
    decoder = args.decoder(args.input_dim, args.hid_dim, args.y_frames, args.n_layers, args.batch_size,
                           args.dropout, args.use_bn,args.attn_head,args.attn_size, args.x_frames, activation="ReLU")
    encoder.to(args.device)
    decoder.to(args.device)

    encoder.load_state_dict(torch.load(args.innate_path + '\\' + str(site_val_losses) +'_epoch'+'_encoder' + '.pt'))
    decoder.load_state_dict(torch.load(args.innate_path + '\\' + str(site_val_losses) +'_epoch' +'_decoder'+ '.pt'))

    ## graph
    train_val_graph = epoch_graph_list[site_val_losses]

    test_loss_metric1, test_loss_metric2, test_loss_metric3, graph3, unused_test = test(encoder, decoder, partition, args)
    print('MAE: {},\n RMSE: {}, \n MAPE: {}'
          .format(test_loss_metric1, test_loss_metric2, test_loss_metric3))

    with open(args.innate_path + '\\'+ str(site_val_losses)+'Epoch_test_metric' +'.csv', 'w') as fd:
        print('MAE : {} \n RMSE : {} \n MAPE : {}'
              .format(test_loss_metric1, test_loss_metric2, test_loss_metric3), file=fd)
    # ======= Add Result to Dictionary ======= #
    result = {}
    result['train_losses'] = train_losses
    result['val_losses'] = val_losses
    result['test_loss_metric1'] = test_loss_metric1
    result['test_loss_metric2'] = test_loss_metric2
    result['test_loss_metric3'] = test_loss_metric3
    result['train_val_graph'] = train_val_graph
    result['test_graph'] = graph3
    result['unused_data'] = [unused_triain,unused_val,unused_test]

    return vars(args), result      ## vars(args) 1: args에있는 attrubute들을 dictionary 형태로 보길 원한다면 vars 함


# ====== Random Seed Initialization ====== #
seed = 666
np.random.seed(seed)
torch.manual_seed(seed)

parser = argparse.ArgumentParser()
args = parser.parse_args("")  ## ""을 써주는 이유는 터미널이 아니기때문에
args.exp_name = "exp1_lr"
args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

# ====== Data Loading ====== #
args.symbol = '^KQ11'
args.batch_size = 128
args.x_frames = 20
args.y_frames = 1
args.encoder = LSTM_enc
args.decoder = LSTM_dec

# ====== Model Capacity ===== #
args.input_dim = 1
args.hid_dim = 10
args.n_layers = 1

# ====== Regularization ======= #
args.l2 = 0.00001
args.dropout = 0.0
args.use_bn = True

# ====== Optimizer & Training ====== #
args.optim = 'Adam'  # 'RMSprop' #SGD, RMSprop, ADAM...
args.lr = 0.0001
args.epoch = 250
args.split = 2
# ====== Experiment Variable ====== #
args.attn_head = 3
args.attn_size = 9

args.save_file_path = '[typrrCurrent Directory]\\results'

## 원하는 데이터 리스트
data_list = ['^KS11']

# '^KS11' : KOSPI
# '^KQ11' : 코스닥
# '^IXIC' : 나스닥
# '^GSPC' : SNP 500 지수
# '^DJI' : 다우존수 산업지수
# '^HSI' : 홍콩 항생 지수
# '^N225' : 니케이지수
# '^GDAXI' : 독일 DAX
# '^FTSE' : 영국 FTSE
# '^FCHI' : 프랑스 CAC
# '^IBEX' : 스페인 IBEX
# '^TWII' : 대만 기권
# '^AEX' : 네덜란드 AEX
# '^BSESN' : 인도 센섹스
# 'RTSI.ME' : 러시아 RTXI
# '^BVSP' : 브라질 보베스파 지수
# 'GC=F' : 금 가격
# 'CL=F' : 원유 가격 (2000/ 8 / 20일 부터 데이터가 있음)
# 'BTC-USD' : 비트코인 암호화폐
# 'ETH-USD' : 이더리움 암호화폐

# example
# data_list = ['^KS11', '^KQ11','^IXIC','^GSPC','^DJI','^HSI',
#              '^N225','^GDAXI','^FCHI','^IBEX','^TWII','^AEX',
#              '^BSESN','^BVSP','GC=F','CL=F', 'BTC-USD','ETH-USD']



with open(args.save_file_path + '\\' + 'Attention_lstm_result_t.csv', 'w', encoding='utf-8', newline='') as f:
    wr = csv.writer(f)

    wr.writerow(["model", "stock","train time",
                 "test_MAE_avg", "test_MAE_std",
                 "test_RMSE_avg", "test_RMSE_std",
                 "test_PAPE_avg", "test_MAPE_std"])

    for j in data_list:
        setattr(args, 'symbol', j)

        model_name = 'attention_LSTM'

        args.new_file_path = args.save_file_path +'\\'+ model_name+'_' + args.symbol
        os.makedirs(args.new_file_path)
        # 데이터 목록 중 시작 범위가 다른 데이터들과 나머지 데이터들
        if args.symbol == '^KQ11':
            data_start = "2013-03-03"
            data_end = "2020-12-31"
        elif args.symbol == 'CL=F':
            data_start = "2011-01-01"      ##(2000, 8, 23) ## 시작
            data_end = "2020-12-31"
        elif args.symbol == 'BTC-USD':
            data_start = "2014-09-17"
            data_end = "2020-12-31"
        elif args.symbol == 'ETH-USD':
            data_start = "2015-08-07"
            data_end = "2020-12-31"
        else:  ## 나머지 모든 데이터들
            data_start = "2011-01-01"
            data_end = "2020-12-31"

        ts = time.time()
        splitted_test_train = Data_Spliter_CrossVal(args.symbol, data_start, data_end, n_splits=args.split)
        entire_data = splitted_test_train.entire_data()

        args.series_Data = splitted_test_train.entire_data
        test_metric1_list = []
        test_metric2_list = []
        test_metric3_list = []
        for iteration_n in range(args.split):
            args.iteration = iteration_n
            train_data, test_data = splitted_test_train[args.iteration][0], splitted_test_train[args.iteration][1]
            test_size = splitted_test_train.test_size
            splitted_train_val = Train_Data_Spliter_CrossVal(train_data, args.symbol, test_size=test_size)
            train_data, val_data = splitted_train_val[1][0], splitted_train_val[1][1]

            trainset = StockDataset(train_data, args.x_frames, args.y_frames)
            valset   = StockDataset(val_data, args.x_frames, args.y_frames)
            testset  = StockDataset(test_data, args.x_frames, args.y_frames)
            partition = {'train': trainset, 'val': valset, 'test': testset}

            args.innate_path = args.new_file_path + '\\' + str(args.iteration) +'_iter' ## 내부 파일경로
            os.makedirs(args.innate_path)
            print(args)

            setting, result = experiment(partition, deepcopy(args))
            test_metric1_list.append(result['test_loss_metric1'])
            test_metric2_list.append(result['test_loss_metric2'])
            test_metric3_list.append(result['test_loss_metric3'])

            ## 그림
            fig = plt.figure()
            plt.plot(result['train_losses'])
            plt.plot(result['val_losses'])
            plt.legend(['train_losses', 'val_losses'], fontsize=15)
            plt.xlabel('epoch', fontsize=15)
            plt.ylabel('loss', fontsize=15)
            plt.grid()
            plt.savefig(args.new_file_path + '\\' + str(args.iteration) + '_fig' + '.png')
            plt.close(fig)

            predicted_traing = result['train_val_graph'][0]
            predicted_valg = result['train_val_graph'][1]
            predicted_testg = result['test_graph']
            entire_dataa = entire_data['Close'].values.tolist()

            train_length = len(predicted_traing)
            val_length = len(predicted_valg)
            test_length = len(predicted_testg)
            entire_length = len(entire_dataa)

            unused_triain = result['unused_data'][0]
            unused_val = result['unused_data'][1]
            unused_test = result['unused_data'][2]

            train_index = list(range(args.x_frames,args.x_frames+train_length))
            val_index = list(range(args.x_frames+train_length+unused_triain+args.x_frames, args.x_frames+train_length+unused_triain+args.x_frames+val_length))
            test_index = list(range(args.x_frames+train_length+unused_triain+args.x_frames+val_length+unused_val+args.x_frames, args.x_frames+train_length+unused_triain+args.x_frames+val_length+unused_val+args.x_frames+test_length))
            entire_index = list(range(entire_length))


            fig2 = plt.figure()
            plt.plot(entire_index, entire_dataa)
            plt.plot(train_index, predicted_traing)
            plt.plot(val_index, predicted_valg)
            plt.plot(test_index, predicted_testg)
            plt.legend(['raw_data', 'predicted_train', 'predicted_val','predicted_test'], fontsize=15)
            plt.xlim(0, entire_length)
            plt.xlabel('time', fontsize=15)
            plt.ylabel('value', fontsize=15)
            plt.grid()
            plt.savefig(args.new_file_path + '\\' + str(args.iteration) + '_chart_fig' + '.png')
            plt.close(fig2)


            #save_exp_result(setting, result)

        avg_test_metric1 = sum(test_metric1_list) / len(test_metric1_list)
        avg_test_metric2 = sum(test_metric2_list) / len(test_metric2_list)
        avg_test_metric3 = sum(test_metric3_list) / len(test_metric3_list)
        std_test_metric1 = np.std(test_metric1_list)
        std_test_metric2 = np.std(test_metric2_list)
        std_test_metric3 = np.std(test_metric3_list)

        #csv파일에 기록하기
        te = time.time()


        train_time = te-ts
        #csv파일에 기록하기
        wr.writerow([model_name, args.symbol, train_time, avg_test_metric1, std_test_metric1,
                                                   avg_test_metric2, std_test_metric2,
                                                   avg_test_metric3, std_test_metric3])

        with open(args.new_file_path + '\\' + 'result_t.txt', 'w') as fd:
            print('MAE \n avg: {}, std : {}\n'.format(avg_test_metric1, std_test_metric1), file=fd)
            print('RMSE \n avg: {}, std : {}\n'.format(avg_test_metric2, std_test_metric2), file=fd)
            print('MAPE \n avg: {}, std : {}\n'.format(avg_test_metric3, std_test_metric3), file=fd)
        print('{}_{} mean_MAE : {}'.format(model_name, args.symbol, avg_test_metric1))
        print('{}_{} mean_RMSE : {}'.format(model_name, args.symbol, avg_test_metric2))
        print('{}_{} mean_MAPE : {}'.format(model_name, args.symbol, avg_test_metric3))