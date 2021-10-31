import attention
import torch
import torch.nn as nn


class LSTM_enc(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, batch_size):
        super(LSTM_enc,self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.batch_size = batch_size
        ## 파이토치에 있는 lstm모듈
        ## output dim 은 self.regressor에서 사용됨
        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, self.num_layers)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        return (torch.zeros(self.num_layers, self.batch_size, self.hidden_dim),
                torch.zeros(self.num_layers, self.batch_size, self.hidden_dim))


    def forward(self, x):
        # 새로 opdate 된 self.hidden과 lstm_out을 return 해줌
        # self.hidden 각각의 layer의 모든 hidden state 를 갖고있음
        ## LSTM의 hidden state에는 tuple로 cell state포함, 0번째는 hidden state tensor, 1번째는 cell state
        lstm_out, self.hidden = self.lstm(x, self.hidden)

        return lstm_out, self.hidden



class LSTM_dec(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, batch_size, dropout,use_bn, attn_head,
                 attn_size,x_frames, activation="ReLU"):
        super(LSTM_dec,self).__init__()
        self.input_dim = input_dim
        self.seq_len = x_frames
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.attn_head = attn_head
        self.attn_size = attn_size
        self.dropout = dropout
        self.use_bn = use_bn
        self.activation = getattr(nn, activation)()

        ## 파이토치에 있는 lstm모듈
        ## output dim 은 self.regressor에서 사용됨
        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, self.num_layers)
        self.attention = attention.Attention(self.attn_head, self.attn_size, self.hidden_dim, self.hidden_dim,
                                             self.hidden_dim, self.dropout)


        self.regression_input_size = self.attn_size + self.hidden_dim
        self.regressor = self.make_regressor()

    def make_regressor(self): # 간단한 MLP를 만드는 함수
        layers = []
        if self.use_bn:
            layers.append(nn.BatchNorm1d(self.seq_len * self.regression_input_size))  ##  nn.BatchNorm1d
        layers.append(nn.Dropout(self.dropout))    ##  nn.Dropout

        ## hidden dim을 outputdim으로 바꿔주는 MLP
        layers.append(nn.Linear(self.seq_len *self.regression_input_size, self.output_dim))
        regressor = nn.Sequential(*layers)
        return regressor

    def forward(self, x, encoder_hidden_states):
        # 새로 opdate 된 self.hidden과 lstm_out을 return 해줌
        # self.hidden 각각의 layer의 모든 hidden state 를 갖고있음
        ## LSTM의 hidden state에는 tuple로 cell state포함, 0번째는 hidden state tensor, 1번째는 cell state

        lstm_out, self.hidden = self.lstm(x, encoder_hidden_states)
        lstm_out = lstm_out.transpose(0, 1)
        encoder_hidden_states = encoder_hidden_states[0]
        attn_applied, attn_weights = self.attention(lstm_out, encoder_hidden_states, encoder_hidden_states)


        ## lstm_out : 각 time step에서의 lstm 모델의 output 값
        ## lstm_out[-1] : 맨마지막의 아웃풋 값으로 그 다음을 예측하고싶은 것이기 때문에 -1을 해줌
        concat = self.activation(torch.cat([attn_applied, lstm_out], dim=2))

        y_pred = self.regressor(concat.view(self.batch_size, -1)) ## self.batch size로 reshape해 regressor에 대입

        return y_pred, attn_weights