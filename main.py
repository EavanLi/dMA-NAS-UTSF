#---------------------------------------Time Series Prediction----------------------------------------------------
from processing import *
from utils import *
from population import create_pop
from evaluate import FitnessEvaluate
from crossover import crossover_operation
from mutation import mutation_operation
from local_search import local_search_operation
import torch
import argparse
import shutil
import os
import time
import torch
from torch.autograd import Variable
import warnings
torch.manual_seed(1)
warnings.filterwarnings('ignore')

def parse_arguments():
    parser = argparse.ArgumentParser(description="Specify Params for Experimental Setting of MA-NAS")

    parser.add_argument('--evaluation_times', type=int, help="Specify the number of evolution iterations")
    parser.add_argument('--pop_size', type=int)
    parser.add_argument('--alpha', type=float, help="Index of adaptive neighborhood size")
    parser.add_argument('--gamma', type=float, help="Segmentation threshold for the adaptive neighborhood size")
    parser.add_argument('--p_c', type=float, help='Crossover probability')
    parser.add_argument('--p_m', type=float, help='Mutation probability')
    parser.add_argument('--search_modules', type=list, default=['bidirection', 'hidden_size', 'layers_num', 'loss_weights', 'net_type'], help='Available searching modules')

    #parser.add_argument('--time_series_name', type=str, default='temperatures', help="Name of the dataset")
    parser.add_argument('--train_rate', type=float, help="Rate of the training set")
    parser.add_argument('--valid_rate', type=float, help="Rate of the validation set")
    parser.add_argument('--test_rate', type=float, help="Rate of the test set")
    parser.add_argument('--n_in', type=int, help="Length of the input data")
    # parser.add_argument('--n_out', type=int, default=1, help="Length of the output data")
    parser.add_argument('--scaler', type=int, default=None, help="the sclaer")

    parser.add_argument('--decompose_way', type=str, default='EMD', help="Decompose way the dataset") # 'EMD', 'EEMD', 'RobustSTL'

    parser.add_argument('--predict_timestep', type=str, default='mean', help="Weight distribution of the error of each time stamp")

    parser.add_argument('--net', type=list, default=['RNN', 'GRU', 'LSTM', 'CNN'], help="Available network type")
    parser.add_argument('--rnn_layers_max', type=int, default=5, help="Maximum number of hidden layers in Recurrent networks")
    parser.add_argument('--rnn_layers_min', type=int, default=1, help="Minimum number of hidden layers in Recurrent networks")
    parser.add_argument('--rnn_hiddens_max', type=int, default=20, help="Maximum number of hidden nodes in each layer of Recurrent networks")
    parser.add_argument('--rnn_hiddens_min', type=int, default=5, help="Minimum number of hidden nodes in each layer of Recurrent networks")
    parser.add_argument('--rnn_bidirection', type=list, default=[True, False], help="Whether bidirectional")
    parser.add_argument('--cnn_layers_max', type=int, default=5, help="Maximum number of layers in CNN")
    parser.add_argument('--cnn_layers_min', type=int, default=1, help="Minimum number of layers in CNN")
    parser.add_argument('--cnn_kernels_max', type=int, default=3, help="Maximum number of kernel size in each layer of CNN")
    parser.add_argument('--cnn_kernels_min', type=int, default=1, help="Minimum number of kernel size in each layer of CNN")

    parser.add_argument('--weights_loss', type=list, default=[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0], help="Available weight value")

    parser.add_argument('--lr', type=float, help="Learning rate")
    parser.add_argument('--Epochs', type=int, help="Total epoch of training")

    return parser.parse_args()


def main(n_out_perrun, data):
    args = parse_arguments()
    gen_no, eval_time = 0, 0

    args.n_out = n_out_perrun
    args.time_series_name = data
    args.time_series_name = args.time_series_name+'_nout' + str(args.n_out)

    assert args.n_in > args.cnn_kernels_max

    try:
        shutil.rmtree(str(args.time_series_name))
    except:
        pass
    os.mkdir(str(args.time_series_name))
    os.mkdir(str(args.time_series_name)+'/checkpoints')
    os.mkdir(str(args.time_series_name) + '/log')
    os.mkdir(str(args.time_series_name) + '/scripts')


    y = get_time_series(args.time_series_name)

    train_valid_y, test_y = split_test(y, args.train_rate + args.valid_rate)

    scaled_train_valid_y, scaler_train_valid = scaling(train_valid_y, args.time_series_name)
    scaled_test_y, args.scaler = scaling(test_y, args.time_series_name)

    decomposited_train_valid_y = decompose(scaled_train_valid_y, args.decompose_way)
    args.dec_num = len(decomposited_train_valid_y)
    print('The number of components on train_valid sets after decomposition is: %d' % (args.dec_num))
    decomposited_test_y = decompose(scaled_test_y, args.decompose_way)
    print('The number of components on test sets after decomposition is: %d' % (len(decomposited_test_y)))

    decomposited_train_y, decomposited_valid_y = split_valid(decomposited_train_valid_y, args.train_rate, args.valid_rate)
    supervised_train_y = series_to_supervised(decomposited_train_y, args.n_in, args.n_out)
    supervised_valid_y = series_to_supervised(decomposited_valid_y, args.n_in, args.n_out)
    supervised_test_y = series_to_supervised(decomposited_test_y, args.n_in, args.n_out)

    train_dec_x, train_dec_y = split_x_y(supervised_train_y, args.n_in, args.n_out)
    valid_dec_x, valid_dec_y = split_x_y(supervised_valid_y, args.n_in, args.n_out)
    test_dec_x, test_dec_y = split_x_y(supervised_test_y, args.n_in, args.n_out)

    train_org = scaled_train_valid_y[:len(decomposited_train_y[0])]
    valid_org = scaled_train_valid_y[len(decomposited_train_y[0]):]
    test_org = scaled_test_y
    train_org_x, train_org_y = split_x_y(series_to_supervised([train_org], args.n_in, args.n_out), args.n_in, args.n_out)
    valid_org_x, valid_org_y = split_x_y(series_to_supervised([valid_org], args.n_in, args.n_out), args.n_in, args.n_out)
    test_org_x, test_org_y = split_x_y(series_to_supervised([test_org], args.n_in, args.n_out), args.n_in, args.n_out)
    train_dec_org_x = Variable(torch.cat([torch.Tensor(train_dec_x), torch.Tensor(train_org_x)], dim=0).to(device='cuda'))
    train_dec_org_y = Variable(torch.cat([torch.Tensor(train_dec_y), torch.Tensor(train_org_y), torch.Tensor(train_org_y)], dim=0).to(device='cuda'))

    print('-----------------------------------Gen: ' + str(gen_no) + ' | ' + str(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())) + '-----------------------------------')
    print('Ready to initialize the population...')
    pop_class = create_pop(args, gen_no)
    pop_class.initialize_population()

    print('Ready to evaluate the population...')
    fit_class = FitnessEvaluate(pop_class.pop, args)
    fit_class.generate_to_python_file()
    fit_class.evaluate(args, train_dec_org_x, train_dec_org_y, valid_dec_x, valid_dec_y, valid_org_x, valid_org_y, test_dec_x, test_dec_y, test_org_x, test_org_y)
    eval_time += len(pop_class.pop)

    obj_evo, obj_gen_no = [], []
    with open(args.time_series_name+"/log/evolution_info.txt", "a") as f:
        f.write('-----------------------------------dataset: ' + str(args.time_series_name) + '-----------------------------------' + '\n')
        f.write('-----------------------------------Gen: ' + str(gen_no) + ' | ' + str(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())) + '-----------------------------------' + '\n')
        f.write('Population information:' + '\n')
        for indi in pop_class.pop:
            f.write('***' + indi.indi_no + '\n')
            f.write('    net_type: '.ljust(20) + str(indi.net_type) + '\n')
            f.write('    loss_weights: '.ljust(20) + str(indi.loss_weights) + '\n')
            f.write('    rnn_hidden_size: '.ljust(20) + str(indi.rnn_hidden_size) + '\n')
            f.write('    rnn_layers_num: '.ljust(20) + str(indi.rnn_layers_num) + '\n')
            f.write('    rnn_bidirection: '.ljust(20) + str(indi.rnn_bidirection) + '\n')
            f.write('    cnn_layers_num: '.ljust(20) + str(indi.cnn_layers_num) + '\n')
            f.write('    cnn_kernel_size: '.ljust(20) + str(indi.cnn_kernel_size) + '\n')
            f.write('    objective: '.ljust(20) + str(indi.obj) + '\n')
            obj_gen_no.append(indi.obj)
    obj_evo.append(obj_gen_no)


    pop = pop_class.pop
    # GA
    while eval_time < args.evaluation_times:
        gen_no += 1
        print('-----------------------------------Gen: %s' % (gen_no) + ' | ' + str(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())) + '-----------------------------------')

        print('Ready to crossover...')
        crossover_class = crossover_operation(pop, args, gen_no)
        pop_c = crossover_class.do_crossover()

        print('Ready to mutation...')
        mutation_class = mutation_operation(pop_c, args, gen_no)
        pop_m = mutation_class.do_mutation()

        print('Ready to evaluate the mutated population...')
        fit_class = FitnessEvaluate(pop_m, args)
        fit_class.generate_to_python_file()
        fit_class.evaluate(args, train_dec_org_x, train_dec_org_y, valid_dec_x, valid_dec_y, valid_org_x, valid_org_y, test_dec_x, test_dec_y, test_org_x, test_org_y)
        eval_time += len(pop_m)
        print('Evaluation times: %d | %d' % (eval_time, args.evaluation_times))

        with open(args.time_series_name+"/log/evolution_info.txt", "a") as f:
            f.write('Genetic operators information:' + '\n')
            for indi in pop_m:
                f.write('***' + indi.indi_no + '\n')
                f.write('    net_type: '.ljust(20) + str(indi.net_type) + '\n')
                f.write('    loss_weights: '.ljust(20) + str(indi.loss_weights) + '\n')
                f.write('    rnn_hidden_size: '.ljust(20) + str(indi.rnn_hidden_size) + '\n')
                f.write('    rnn_layers_num: '.ljust(20) + str(indi.rnn_layers_num) + '\n')
                f.write('    rnn_bidirection: '.ljust(20) + str(indi.rnn_bidirection) + '\n')
                f.write('    cnn_layers_num: '.ljust(20) + str(indi.cnn_layers_num) + '\n')
                f.write('    cnn_kernel_size: '.ljust(20) + str(indi.cnn_kernel_size) + '\n')
                f.write('    objective: '.ljust(20) + str(indi.obj) + '\n')


        #a discriminative local search operation
        local_search_class = local_search_operation(pop_m, args, gen_no)
        pop_ls, pop_BIR = local_search_class.do_local_search(train_dec_org_x, train_dec_org_y, valid_dec_x, valid_dec_y, valid_org_x, valid_org_y, test_dec_x, test_dec_y, test_org_x, test_org_y)
        pop_ls = local_search_class.eva_local_search(train_dec_org_x, train_dec_org_y, valid_dec_x, valid_dec_y, valid_org_x, valid_org_y, test_dec_x, test_dec_y, test_org_x, test_org_y)
        eval_time += args.dec_num + 1 + len(pop_ls)
        print('Evaluation times: %d | %d' % (eval_time, args.evaluation_times))

        print('Ready for environmental selection...')
        env_sel_class = environment_selection(pop, pop_m, pop_ls, pop_BIR, args, gen_no)
        pop = env_sel_class.do_env_sel()

        obj_gen_no = []
        with open(args.time_series_name+"/log/evolution_info.txt", "a") as f:
            f.write('Environmental selection information:' + '\n')
            for indi in pop:
                f.write('***' + indi.indi_no + '\n')
                f.write('    net_type: '.ljust(20) + str(indi.net_type) + '\n')
                f.write('    loss_weights: '.ljust(20) + str(indi.loss_weights) + '\n')
                f.write('    rnn_hidden_size: '.ljust(20) + str(indi.rnn_hidden_size) + '\n')
                f.write('    rnn_layers_num: '.ljust(20) + str(indi.rnn_layers_num) + '\n')
                f.write('    rnn_bidirection: '.ljust(20) + str(indi.rnn_bidirection) + '\n')
                f.write('    cnn_layers_num: '.ljust(20) + str(indi.cnn_layers_num) + '\n')
                f.write('    cnn_kernel_size: '.ljust(20) + str(indi.cnn_kernel_size) + '\n')
                f.write('    objective: '.ljust(20) + str(indi.obj) + '\n')
                obj_gen_no.append(indi.obj)
        obj_evo.append(obj_gen_no)

    print('dMA-NAS-UTSF is finished.')


if __name__ == "__main__":
    # taking 'UCR_insectEPG5_177' with L_y=[1,6,48,96] (i.e., n_out=[1,6,48,96]) as an example
    for n_out in [1, 6, 48, 96]:
        for data in ['UCR_insectEPG5_177']:
            main(n_out, data)
