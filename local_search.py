# ---------------------------------------------a discriminative local search operator------------------------------------

import copy
import random
from evaluate import FitnessEvaluate
from sklearn import preprocessing
import numpy as np

class local_search_operation(object):
    def __init__(self, pop_m, args, gen_no):
        self.pop_m = pop_m
        self.args = args
        self.gen_no = gen_no
        self.pop_ls = None
        self.pop_BIR = []

    def do_local_search(self, train_dec_org_x, train_dec_org_y, valid_dec_x, valid_dec_y, valid_org_x, valid_org_y, test_dec_x, test_dec_y, test_org_x, test_org_y):
        self.pop_ls = []

        fit_pop_m = []
        for i in range(len(self.pop_m)):
            fit_pop_m.append(self.pop_m[i].obj)
        i = fit_pop_m.index(min(fit_pop_m))

        print('Ready to weight perturbation...')
        pop_BIR_i = []
        for brach_num in range(self.args.dec_num + 1):
            parent = copy.deepcopy(self.pop_m[i])
            parent.loss_weights[brach_num] = parent.loss_weights[brach_num] + 0.1
            parent.indi_no = parent.indi_no + str(brach_num).zfill(2)
            parent.obj = -1.0
            pop_BIR_i.append(parent)

        print('Ready to evaluate the population of ' + self.pop_m[i].indi_no + ' after the weight perturbation...')
        fit_class = FitnessEvaluate(pop_BIR_i, self.args)
        fit_class.generate_to_python_file()
        fit_class.evaluate(self.args, train_dec_org_x, train_dec_org_y, valid_dec_x, valid_dec_y, valid_org_x, valid_org_y, test_dec_x, test_dec_y, test_org_x, test_org_y)

        with open(self.args.time_series_name+"/log/evolution_info.txt", "a") as f:
            f.write('BIR information:' + '\n')
            for indi in pop_BIR_i:
                f.write('***' + indi.indi_no + '\n')
                f.write('    net_type: '.ljust(20) + str(indi.net_type) + '\n')
                f.write('    loss_weights: '.ljust(20) + str(indi.loss_weights) + '\n')
                f.write('    rnn_hidden_size: '.ljust(20) + str(indi.rnn_hidden_size) + '\n')
                f.write('    rnn_layers_num: '.ljust(20) + str(indi.rnn_layers_num) + '\n')
                f.write('    rnn_bidirection: '.ljust(20) + str(indi.rnn_bidirection) + '\n')
                f.write('    cnn_layers_num: '.ljust(20) + str(indi.cnn_layers_num) + '\n')
                f.write('    cnn_kernel_size: '.ljust(20) + str(indi.cnn_kernel_size) + '\n')
                f.write('    objective: '.ljust(20) + str(indi.obj) + '\n')

        fit_BIR, parent_obj, Diff_fit_BIR = [], [], []
        for brach_num in range(self.args.dec_num + 1):
            fit_BIR.append(pop_BIR_i[brach_num].obj)
            parent_obj.append(self.pop_m[i].obj)
            Diff_fit_BIR.append(self.pop_m[i].obj - pop_BIR_i[brach_num].obj)

        scaler = preprocessing.MaxAbsScaler()
        Diff_fit_BIR_scaled = scaler.fit_transform(np.array([Diff_fit_BIR]).reshape(-1,1))

        neighborhood_size = self.cal_neighborhood_size(Diff_fit_BIR_scaled)
        print('Ready to do local search for each branch of ' + self.pop_m[i].indi_no + ' | neighborhood size : ' + str(neighborhood_size))


        for brach_num in range(self.args.dec_num + 1):
            pop_ls_i_branch = self.ls_branch(brach_num, pop_BIR_i[brach_num], neighborhood_size[brach_num])
            for indi_ls in pop_ls_i_branch:
                self.pop_ls.append(indi_ls)
        for indi_BIR in pop_BIR_i:
            self.pop_BIR.append(indi_BIR)

        return self.pop_ls, self.pop_BIR

    def eva_local_search(self, train_dec_org_x, train_dec_org_y, valid_dec_x, valid_dec_y, valid_org_x, valid_org_y, test_dec_x, test_dec_y, test_org_x, test_org_y):
        print('Ready to evaluate the population after local search...')
        fit_class = FitnessEvaluate(self.pop_ls, self.args)
        fit_class.generate_to_python_file()
        fit_class.evaluate(self.args, train_dec_org_x, train_dec_org_y, valid_dec_x, valid_dec_y, valid_org_x, valid_org_y, test_dec_x, test_dec_y, test_org_x, test_org_y)

        with open(self.args.time_series_name+"/log/evolution_info.txt", "a") as f:
            f.write('Local search information:' + '\n')
            for indi in self.pop_ls:
                f.write('***' + indi.indi_no + '\n')
                f.write('    net_type: '.ljust(20) + str(indi.net_type) + '\n')
                f.write('    loss_weights: '.ljust(20) + str(indi.loss_weights) + '\n')
                f.write('    rnn_hidden_size: '.ljust(20) + str(indi.rnn_hidden_size) + '\n')
                f.write('    rnn_layers_num: '.ljust(20) + str(indi.rnn_layers_num) + '\n')
                f.write('    rnn_bidirection: '.ljust(20) + str(indi.rnn_bidirection) + '\n')
                f.write('    cnn_layers_num: '.ljust(20) + str(indi.cnn_layers_num) + '\n')
                f.write('    cnn_kernel_size: '.ljust(20) + str(indi.cnn_kernel_size) + '\n')
                f.write('    objective: '.ljust(20) + str(indi.obj) + '\n')

        return self.pop_ls

    def cal_neighborhood_size(self, fit_BIR_scaled):
        neighborhood_size = []
        for brach_num in range(self.args.dec_num + 1):
            if fit_BIR_scaled[brach_num] < -self.args.gamma:
                neighborhood_size.append(1)
            elif fit_BIR_scaled[brach_num] >= -self.args.gamma and fit_BIR_scaled[brach_num] <= self.args.gamma:
                neighborhood_size.append(0)
            else:
                neighborhood_size.append(int(self.args.alpha**(fit_BIR_scaled[brach_num]-self.args.gamma)))

        return neighborhood_size

    def ls_branch(self, branch_num, indi, ls_size):
        pop_ls_i_branch = []

        if ls_size == 0:
            pass
        elif ls_size == 1:
            parent = copy.deepcopy(indi)

            mutation_value = random.choice(self.args.net)
            while mutation_value == parent.net_type[branch_num]:
                mutation_value = random.choice(self.args.net)
            parent.net_type[branch_num] = mutation_value

            if mutation_value == 'CNN':
                parent.rnn_hidden_size[branch_num] = None
                parent.rnn_layers_num[branch_num] = None
                parent.rnn_bidirection[branch_num] = None
                parent.cnn_kernel_size[branch_num] = random.randint(self.args.cnn_kernels_min, self.args.cnn_kernels_max)
                parent.cnn_layers_num[branch_num] = random.randint(self.args.cnn_layers_min, min(self.args.cnn_layers_max, int(self.args.n_in / parent.cnn_kernel_size[branch_num])))
            else:
                parent.cnn_kernel_size[branch_num] = None
                parent.cnn_layers_num[branch_num] = None
                parent.rnn_hidden_size[branch_num] = random.randint(self.args.rnn_hiddens_min, self.args.rnn_hiddens_max)
                parent.rnn_layers_num[branch_num] = random.randint(self.args.rnn_layers_min, self.args.rnn_layers_max)
                parent.rnn_bidirection[branch_num] = random.choice(self.args.rnn_bidirection)
            parent.indi_no = parent.indi_no + '00'
            parent.obj = -1.0
            pop_ls_i_branch.append(parent)

        else:
            ls_num = 0

            parent = copy.deepcopy(indi)
            mutation_value = random.choice(self.args.net)
            while mutation_value == parent.net_type[branch_num]:
                mutation_value = random.choice(self.args.net)
            parent.net_type[branch_num] = mutation_value
            if mutation_value == 'CNN':
                parent.rnn_hidden_size[branch_num] = None
                parent.rnn_layers_num[branch_num] = None
                parent.rnn_bidirection[branch_num] = None
                parent.cnn_kernel_size[branch_num] = random.randint(self.args.cnn_kernels_min, self.args.cnn_kernels_max)
                parent.cnn_layers_num[branch_num] = random.randint(self.args.cnn_layers_min, min(self.args.cnn_layers_max, int(self.args.n_in / parent.cnn_kernel_size[branch_num])))
            else:
                parent.cnn_kernel_size[branch_num] = None
                parent.cnn_layers_num[branch_num] = None
                parent.rnn_hidden_size[branch_num] = random.randint(self.args.rnn_hiddens_min, self.args.rnn_hiddens_max)
                parent.rnn_layers_num[branch_num] = random.randint(self.args.rnn_layers_min, self.args.rnn_layers_max)
                parent.rnn_bidirection[branch_num] = random.choice(self.args.rnn_bidirection)
            parent.indi_no = parent.indi_no + str(ls_num).zfill(2)
            parent.obj = -1.0
            pop_ls_i_branch.append(parent)
            ls_num += 1

            while ls_num < ls_size:
                parent = copy.deepcopy(indi)
                if parent.net_type[branch_num] == 'CNN':
                    parent.cnn_kernel_size[branch_num] = random.randint(self.args.cnn_kernels_min, self.args.cnn_kernels_max)
                    parent.cnn_layers_num[branch_num] = random.randint(self.args.cnn_layers_min, min(self.args.cnn_layers_max, int(self.args.n_in / parent.cnn_kernel_size[branch_num])))
                else:
                    parent.rnn_hidden_size[branch_num] = random.randint(self.args.rnn_hiddens_min, self.args.rnn_hiddens_max)
                    parent.rnn_layers_num[branch_num] = random.randint(self.args.rnn_layers_min, self.args.rnn_layers_max)
                    parent.rnn_bidirection[branch_num] = random.choice(self.args.rnn_bidirection)
                parent.indi_no = parent.indi_no + str(ls_num).zfill(2)
                parent.obj = -1.0
                pop_ls_i_branch.append(parent)
                ls_num += 1

        return pop_ls_i_branch
