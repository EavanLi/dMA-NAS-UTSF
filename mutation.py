import copy
import random

class mutation_operation(object):
    def __init__(self, pop_c, args, gen_no):
        self.pop_c = pop_c
        self.args = args
        self.gen_no = gen_no
        self.pop_m = []

    def do_mutation(self):
        with open(self.args.time_series_name+"/log/evolution_info.txt", "a") as f:
            f.write('Mutation information:' + '\n')
        for i in range(len(self.pop_c)):
            parent = copy.deepcopy(self.pop_c[i])
            if random.random() < self.args.p_m:
                pos = random.randint(0, len(parent.rnn_bidirection) - 1)

                mutation_value = random.choice(self.args.weights_loss)
                while mutation_value == parent.loss_weights[pos]:
                    mutation_value = random.choice(self.args.weights_loss)
                parent.loss_weights[pos] = mutation_value

                mutation_value = random.choice(self.args.net)
                while mutation_value == parent.net_type[pos]:
                    mutation_value = random.choice(self.args.net)
                parent.net_type[pos] = mutation_value

                if mutation_value == 'CNN':
                    parent.rnn_hidden_size[pos] = None
                    parent.rnn_layers_num[pos] = None
                    parent.rnn_bidirection[pos] = None
                    parent.cnn_kernel_size[pos] = random.randint(self.args.cnn_kernels_min, self.args.cnn_kernels_max)
                    parent.cnn_layers_num[pos] = random.randint(self.args.cnn_layers_min, min(self.args.cnn_layers_max, int(self.args.n_in/parent.cnn_kernel_size[pos])))
                else:
                    parent.cnn_kernel_size[pos] = None
                    parent.cnn_layers_num[pos] = None
                    parent.rnn_hidden_size[pos] = random.randint(self.args.rnn_hiddens_min, self.args.rnn_hiddens_max)
                    parent.rnn_layers_num[pos] = random.randint(self.args.rnn_layers_min, self.args.rnn_layers_max)
                    parent.rnn_bidirection[pos] = random.choice(self.args.rnn_bidirection)

                with open(self.args.time_series_name+"/log/evolution_info.txt", "a") as f:
                    f.write('    Mutation parents: '.ljust(25) + str(self.pop_c[i].indi_no) + '\n')
                    f.write('    Mutation position: '.ljust(25) + str(pos) + '\n')

            self.pop_m.append(parent)

        return self.pop_m
