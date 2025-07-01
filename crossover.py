#----------------------------------------------------------------------------------------------------
import time
from utils import *
import random
import copy

class crossover_operation(object):
    def __init__(self, pop, args, gen_no):
        self.pop = pop
        self.args = args
        self.gen_no = gen_no
        self.pop_c = []

    def do_crossover(self):
        with open(self.args.time_series_name+"/log/evolution_info.txt", "a") as f:
            f.write('--------------------------Gen: %s' % (self.gen_no) + ' | ' + str(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())) + '--------------------------' + '\n')
            f.write('Crossover information:' + '\n')
        for _ in range(len(self.pop)//2):
            if random.random() < self.args.p_c:

                indi1, indi2 = tournament_selection(self.pop)
                parent1, parent2 = copy.deepcopy(self.pop[indi1]), copy.deepcopy(self.pop[indi2])

                parent1.obj, parent2.obj = -1.0, -1.0
                parent1.indi_no = 'indi%02d%02d' % (self.gen_no, len(self.pop_c))
                parent2.indi_no = 'indi%02d%02d' % (self.gen_no, len(self.pop_c)+1)


                pos = random.randint(1, len(parent1.net_type) - 1)
                temp_net_type = parent1.net_type[pos:]
                temp_loss_weights = parent1.loss_weights[pos:]
                temp_rnn_hidden_size = parent1.rnn_hidden_size[pos:]
                temp_rnn_layers_num = parent1.rnn_layers_num[pos:]
                temp_rnn_bidirection = parent1.rnn_bidirection[pos:]
                temp_cnn_layers_num = parent1.cnn_layers_num[pos:]
                temp_cnn_kernel_size = parent1.cnn_kernel_size[pos:]
                parent1.net_type[pos:] = parent2.net_type[pos:]
                parent1.loss_weights[pos:] = parent2.loss_weights[pos:]
                parent1.rnn_hidden_size[pos:] = parent2.rnn_hidden_size[pos:]
                parent1.rnn_layers_num[pos:] = parent2.rnn_layers_num[pos:]
                parent1.rnn_bidirection[pos:] = parent2.rnn_bidirection[pos:]
                parent1.cnn_layers_num[pos:] = parent2.cnn_layers_num[pos:]
                parent1.cnn_kernel_size[pos:] = parent2.cnn_kernel_size[pos:]
                parent2.net_type[pos:] = temp_net_type
                parent2.loss_weights[pos:] = temp_loss_weights
                parent2.rnn_hidden_size[pos:] = temp_rnn_hidden_size
                parent2.rnn_layers_num[pos:] = temp_rnn_layers_num
                parent2.rnn_bidirection[pos:] = temp_rnn_bidirection
                parent2.cnn_layers_num[pos:] = temp_cnn_layers_num
                parent2.cnn_kernel_size[pos:] = temp_cnn_kernel_size

                with open(self.args.time_series_name+"/log/evolution_info.txt", "a") as f:
                    f.write('    Crossover parents: '.ljust(25)+str(self.pop[indi1].indi_no)+' and '+str(self.pop[indi2].indi_no)+' | Crossover offspring : '+str(parent1.indi_no)+' and '+str(parent2.indi_no)+'\n')
                    f.write('    Crossover position: '.ljust(25) + str(pos) + '\n')


                self.pop_c.append(parent1)
                self.pop_c.append(parent2)

        return self.pop_c