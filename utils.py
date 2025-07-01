import numpy as np
import random
import math

class environment_selection(object):
    def __init__(self, pop, pop_m, pop_ls, pop_BIR, args, gen_no):
        self.pop_m = pop_m
        self.pop_ls = pop_ls
        self.pop_BIR = pop_BIR
        self.pop = pop
        self.args = args
        self.gen_no = gen_no

    def do_env_sel(self):
        combine_pop = self.pop_m + self.pop + self.pop_ls + self.pop_BIR
        combine_obj = []
        for indi in combine_pop:
            combine_obj.append(indi.obj)
        sorted_id = sorted(range(len(combine_obj)), key=lambda k: combine_obj[k], reverse=False) # 从小到大排序
        offspring = []
        for id in sorted_id[:self.args.pop_size]:
            offspring.append(combine_pop[id])

        for i in range(len(offspring)):
            offspring[i].indi_no = 'indi%02d%02d' % (self.gen_no, i)

        return offspring

def tournament_selection(pop):
    indi1 = choose_one_parent(pop)
    indi2 = choose_one_parent(pop)
    while indi1 == indi2:
        indi2 = choose_one_parent(pop)
    assert indi1 < len(pop)
    assert indi2 < len(pop)

    return indi1, indi2

def choose_one_parent(pop):
    count_ = len(pop)
    idx1 = int(np.floor(np.random.random() * count_))
    idx2 = int(np.floor(np.random.random() * count_))
    while idx2 == idx1:
        idx2 = int(np.floor(np.random.random() * count_))

    if pop[idx1].obj < pop[idx1].obj:
        return idx1
    else:
        return idx2


def evaluate(predict_y, test_org_y):
    predict_MSE = []
    for timestep in range(predict_y.shape[1]):
        MSE_timestep = 0
        for sample_num in range(predict_y.shape[0]):
            MSE_timestep += (predict_y[sample_num][timestep] - test_org_y[sample_num][timestep]) ** 2
        MSE_timestep = MSE_timestep / predict_y.shape[0]
        predict_MSE.append(MSE_timestep)
    return predict_MSE

def cal_RMSE(predict, test_y, n_out):
    predict_RMSE = []
    for timestep in range(n_out):
        RMSE_timestep = 0
        for sample_num in range(len(predict)):
            RMSE_timestep += (predict[sample_num][timestep] - test_y[sample_num][timestep]) ** 2
        RMSE_timestep = math.sqrt(RMSE_timestep / len(predict))
        predict_RMSE.append(RMSE_timestep)
    return predict_RMSE

def cal_MAE(predict, test_y, n_out):
    predict_MAE = []
    for timestep in range(n_out):
        MAE_timestep = 0
        for sample_num in range(len(predict)):
            MAE_timestep += abs(predict[sample_num][timestep] - test_y[sample_num][timestep])
        MAE_timestep = MAE_timestep / len(predict)
        predict_MAE.append(MAE_timestep)
    return predict_MAE

def extract_obj(evolution_info, pop_size):
    obj_evo = []
    for i in range(len(evolution_info)):
        obj_gen_no = []
        if 'Population information' in evolution_info[i] or 'Environmental selection information' in evolution_info[i]:
            for pop in range(pop_size):
                obj_gen_no.append(float(evolution_info[i+(pop+1)*9][20:]))
            obj_evo.append(obj_gen_no)

    return obj_evo

def extract_loss_weights(evolution_info, pop_size):
    loss_weights = []
    for i in range(len(evolution_info)):
        obj_gen_no = []
        if 'Population information' in evolution_info[i] or 'Environmental selection information' in evolution_info[i]:
            for pop in range(pop_size):
                loss_pop = transform_to_list(evolution_info[i+(pop+1)*9-6][20:], module = 'loss_weights')
                obj_gen_no.append(loss_pop)
            loss_weights.append(obj_gen_no)
    return loss_weights

def transform_to_list(string_info, module):
    string_info = string_info.strip()[1:-1].split(",")
    list_info = []
    for ele in string_info:
        if module == 'loss_weights':
            list_info.append(float(ele))
        elif module == 'net_type':
            list_info.append(ele.strip()[1:-1])
        elif module == 'bidirection':
            list_info.append(ele.strip())
        elif module == 'rnn_hidden_size':
            if 'None' not in ele:
                list_info.append(int(ele))
        elif module == 'rnn_layers_num':
            if 'None' not in ele:
                list_info.append(int(ele))
        elif module == 'cnn_layers_num':
            if 'None' not in ele:
                list_info.append(int(ele))
        elif module == 'cnn_kernel_size':
            if 'None' not in ele:
                list_info.append(int(ele))
    return list_info

def extract_net_type(evolution_info, pop_size):
    net_type = []
    for i in range(len(evolution_info)):
        net_gen_no = []
        if 'Population information' in evolution_info[i] or 'Environmental selection information' in evolution_info[i]:
            for pop in range(pop_size):
                net_pop = transform_to_list(evolution_info[i+(pop+1)*9-7][20:], module = 'net_type')
                net_gen_no.append(net_pop)
            net_type.append(net_gen_no)
    return net_type

def extract_bidirection(evolution_info, pop_size):
    bidirection = []
    for i in range(len(evolution_info)):
        bidirection_gen_no = []
        if 'Population information' in evolution_info[i] or 'Environmental selection information' in evolution_info[i]:
            for pop in range(pop_size):
                bidirection_pop = transform_to_list(evolution_info[i+(pop+1)*9-3][20:], module = 'bidirection')
                bidirection_gen_no.append(bidirection_pop)
            bidirection.append(bidirection_gen_no)
    return bidirection

def extract_rnn_hidden_size(evolution_info, pop_size):
    rnn_hidden_size = []
    for i in range(len(evolution_info)):
        rnn_hidden_size_gen_no = []
        if 'Population information' in evolution_info[i] or 'Environmental selection information' in evolution_info[i]:
            for pop in range(pop_size):
                loss_pop = transform_to_list(evolution_info[i+(pop+1)*9-5][20:], module = 'rnn_hidden_size')
                if len(loss_pop) > 0:
                    rnn_hidden_size_gen_no.append(sum(loss_pop)/len(loss_pop))
            rnn_hidden_size.append(rnn_hidden_size_gen_no)
    return rnn_hidden_size

def extract_rnn_layers_num(evolution_info, pop_size):
    rnn_layers_num = []
    for i in range(len(evolution_info)):
        rnn_layers_num_gen_no = []
        if 'Population information' in evolution_info[i] or 'Environmental selection information' in evolution_info[i]:
            for pop in range(pop_size):
                layers_num_pop = transform_to_list(evolution_info[i+(pop+1)*9-4][20:], module = 'rnn_layers_num')
                if len(layers_num_pop) > 0:
                    rnn_layers_num_gen_no.append(sum(layers_num_pop)/len(layers_num_pop))
            rnn_layers_num.append(rnn_layers_num_gen_no)
    return rnn_layers_num

def extract_cnn_layers_num(evolution_info, pop_size):
    cnn_layers_num = []
    for i in range(len(evolution_info)):
        cnn_layers_num_gen_no = []
        if 'Population information' in evolution_info[i] or 'Environmental selection information' in evolution_info[i]:
            for pop in range(pop_size):
                layers_num_pop = transform_to_list(evolution_info[i+(pop+1)*9-2][20:], module = 'cnn_layers_num')
                if len(layers_num_pop) > 0:
                    cnn_layers_num_gen_no.append(sum(layers_num_pop)/len(layers_num_pop))
            cnn_layers_num.append(cnn_layers_num_gen_no)
    return cnn_layers_num

def extract_cnn_kernel_size(evolution_info, pop_size):
    cnn_kernel_size = []
    for i in range(len(evolution_info)):
        cnn_kernel_size_gen_no = []
        if 'Population information' in evolution_info[i] or 'Environmental selection information' in evolution_info[i]:
            for pop in range(pop_size):
                layers_num_pop = transform_to_list(evolution_info[i+(pop+1)*9-1][20:], module = 'cnn_kernel_size')
                if len(layers_num_pop) > 0:
                    cnn_kernel_size_gen_no.append(sum(layers_num_pop)/len(layers_num_pop))
            cnn_kernel_size.append(cnn_kernel_size_gen_no)
    return cnn_kernel_size

