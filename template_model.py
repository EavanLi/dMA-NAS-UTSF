"""
import torch
import torch.nn as nn
from torch.autograd import Variable
import os
import torch.optim as optim
from utils import *
import numpy as np
import time
import pickle

class Model(nn.Module):
    def __init__(self, n_in, n_out, dec_num, rnn_hidden_size, rnn_layers_num):
        super().__init__()

        self.dec_num = dec_num
        self.dense1 = nn.Linear(dec_num + 1, 1)

        #generated_init

    def forward(self, x):
        #generated_forward

        return output


class MFEALoss(nn.Module):
    def __init__(self, loss_weights, n_out, predict_timestep, dec_num):
        super().__init__()
        self.predict_timestep = predict_timestep
        self.n_out = n_out
        self.dec_num = dec_num
        self.loss_weights = loss_weights


    def forward(self, output, train_y):
        # The front of output is the prediction result of the decomposed components, and the end of output is all the prediction results
        # Calculate the prediction error of each decomposition component
        for i in range(self.dec_num+1):  # 对第i个分解分量统计和直接预测的分支计算Loss
            for j in range(self.n_out):  # 对第i个分解分量的第j个时间戳的误差统计
                a = torch.split(output, int(len(output)/(self.dec_num + 2)), dim = 0)[i].t()[j] # 第i个分量的第j个时间戳的预测结果统计
                b = train_y[i].t()[j] # 第i个分量的第j个时间戳的真实值
                if j == 0:
                    loss_j = torch.sub(a, b).pow(2).sum() / len(a)
                elif j == 1:
                    loss_j = torch.cat([loss_j.reshape(1), (torch.sub(a, b).pow(2).sum() / len(a)).reshape(1)], dim = 0)
                else:
                    loss_j = torch.cat([loss_j,(torch.sub(a, b).pow(2).sum() / len(a)).reshape(1)], dim = 0)
            if self.predict_timestep == 'mean':
                loss_j = loss_j.sum() / self.n_out  # 每个时间戳的预测误差求平均

            if i == 0:
                loss_dec = loss_j
            elif i == 1:
                loss_dec = torch.cat((loss_dec.reshape(1), loss_j.reshape(1)), dim = 0)
            else:
                loss_dec = torch.cat((loss_dec, loss_j.reshape(1)), dim = 0)

        # Calculate the prediction error of the final prediction result
        for j in range(self.n_out):
            a = torch.split(output, int(len(output) / (self.dec_num + 2)), dim=0)[-1].t()[j]  # 第i个分量的第j个时间戳的预测结果统计
            b = train_y[-1].t()[j]  # The true value of the j-th timestamp of the i-th component
            if j == 0:
                loss_att = torch.sub(a, b).pow(2).sum() / len(a)
            elif j == 1:
                loss_att = torch.cat([loss_att.reshape(1), (torch.sub(a, b).pow(2).sum() / len(a)).reshape(1)], dim = 0)
            else:
                loss_att = torch.cat([loss_att,(torch.sub(a, b).pow(2).sum() / len(a)).reshape(1)], dim = 0)
        if self.predict_timestep == 'mean':
            loss_att = loss_att.sum() / self.n_out  # 每个时间戳的预测误差求平均

        return torch.matmul(torch.Tensor(self.loss_weights).to(device="cuda"), torch.cat([loss_dec, loss_att.reshape(1)], dim=0))

class RunModel(object):
    def __init__(self, args, indi, train_dec_org_x, train_dec_org_y, valid_dec_x, valid_dec_y, valid_org_x, valid_org_y, test_dec_x, test_dec_y, test_org_x, test_org_y):
        self.indi_no = indi.indi_no
        self.lr = args.lr
        self.loss_weights = indi.loss_weights
        self.n_in = args.n_in
        self.n_out = args.n_out
        self.predict_timestep = args.predict_timestep
        self.dec_num = args.dec_num
        self.Epochs = args.Epochs
        self.scaler = args.scaler
        self.train_dec_org_x = train_dec_org_x
        self.train_dec_org_y = train_dec_org_y
        self.valid_dec_x = valid_dec_x
        self.valid_dec_y = valid_dec_y
        self.valid_org_x = valid_org_x
        self.valid_org_y = valid_org_y
        self.test_dec_x = test_dec_x
        self.test_dec_y = test_dec_y
        self.test_org_x = test_org_x
        self.test_org_y = test_org_y
        self.indi_no = indi.indi_no
        self.rnn_layers_num = indi.rnn_layers_num
        self.rnn_hidden_size = indi.rnn_hidden_size
        self.rnn_bidirection = indi.rnn_bidirection
        self.cnn_layers_num = indi.cnn_layers_num
        self.cnn_kernel_size = indi.cnn_kernel_size
        self.time_series_name = args.time_series_name

    def process(self):
        model = Model(self.n_in, self.n_out, self.dec_num, self.rnn_hidden_size, self.rnn_layers_num).cuda()
        criterion = MFEALoss(self.loss_weights, self.n_out, self.predict_timestep, self.dec_num)
        optimizer = optim.Adam(model.parameters(), lr=self.lr)
        train_loss = []  # 训练过程用分解分量的loss和最终预测的loss，评价阶段只计算最终预测的loss
        valid_loss = []
        valid_loss_best = 10000000 # 记录迄今为止最好的验证集损失
        valid_dec_org_x = Variable(torch.cat([torch.Tensor(self.valid_dec_x), torch.Tensor(self.valid_org_x)], dim=0))
        valid_dec_org_y = Variable(torch.cat([torch.Tensor(self.valid_dec_y), torch.Tensor(self.valid_org_y)], dim=0))
        for epoch in range(1, self.Epochs + 1):
            # 训练集上的计算
            try:
                output_train = model(self.train_dec_org_x)  # model.state_dict()['fc1.weight']
            except RuntimeError as exception:
                if "out of memory" in str(exception):
                    if hasattr(torch.cuda, 'empty_cache'):
                        torch.cuda.empty_cache()
                else:
                    raise exception
            optimizer.zero_grad()
            loss_t = criterion(output_train, self.train_dec_org_y)
            loss_t.backward()
            train_loss.append(loss_t.item())
            optimizer.step()

            # 验证集的计算。每个epoch进行一次验证集的loss计算并保存迄今位置最好的valid_loss的model.pth
            with torch.no_grad():
                output_valid = model(Variable(torch.Tensor(valid_dec_org_x)))
                predict_MSE = evaluate(output_valid[-len(self.valid_org_y[0]):].cpu().detach().numpy(), self.valid_org_y[0])
                predict_MSE_mean = np.mean(predict_MSE)
                valid_loss.append(predict_MSE_mean)
                if predict_MSE_mean < valid_loss_best:
                    with open(self.time_series_name+'/checkpoints/' + self.indi_no + '.pkl', 'wb') as f: # 保存当前model
                        pickle.dump(model, f)
                    valid_loss_best = predict_MSE_mean

        # 测试过程
        with torch.no_grad():
            test_dec_org_x = Variable(torch.cat([torch.Tensor(self.test_dec_x), torch.Tensor(self.test_org_x)], dim=0))
            test_dec_org_y = Variable(torch.cat([torch.Tensor(self.test_dec_y), torch.Tensor(self.test_org_y)], dim=0))
            with open(self.time_series_name+'/checkpoints/' + self.indi_no + '.pkl', 'rb') as f: # 读取验证集上最好的model
                model = pickle.load(f)
            model = model.eval()
            # 输出测试结果
            output_test = model(Variable(torch.Tensor(test_dec_org_x)))
        # 计算每个时间戳的误差 只用最终预测结果计算误差
        RMSE = cal_RMSE(self.scaler.inverse_transform(output_test[-len(self.test_org_y[0]):].cpu().detach().numpy()), self.scaler.inverse_transform(self.test_org_y[0]), self.n_out)
        MAE = cal_MAE(self.scaler.inverse_transform(output_test[-len(self.test_org_y[0]):].cpu().detach().numpy()), self.scaler.inverse_transform(self.test_org_y[0]), self.n_out)

        print('******')
        print(self.indi_no + ': Average prediction error at each time step is ' + str(predict_MSE_mean)[:6])
        print('The Mean RMSE of ' + self.indi_no + ' is: ' + str(sum(RMSE) / self.n_out))
        print('The Mean MAE of ' + self.indi_no + ' is: ' + str(sum(MAE) / self.n_out))


        # 存储训练过程
        with open(self.time_series_name+"/log/%s.txt" % (self.indi_no), "a") as f:
            f.write('-----------------------------------dataset: ' + str(self.time_series_name) + '-----------------------------------' + '\n')
            f.write('-----------------------------------The training process of ' + self.indi_no + ' | ' + str(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())) + '-----------------------------------' +'\n')
            f.write('***The training loss | validation loss:' + '\n')
            for i in range(len(train_loss)):
                f.write('Epoch:' + str(i).zfill(3) + '    training loss: '+ str(train_loss[i]) + ' | ' + '    validation loss: '+ str(valid_loss[i]) + '\n')
            f.write('***The predict_MSE_mean: ' + str(predict_MSE_mean) + '\n')
            f.write('***The Mean RMSE: ' + str(sum(RMSE) / self.n_out) + '\n')
            f.write('***The Mean MAE: ' + str(sum(MAE) / self.n_out) + '\n')

        return predict_MSE_mean
"""