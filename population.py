import random

class create_pop():
    def __init__(self, args, gen_no):
        self.pop = []
        self.pop_size = args.pop_size
        self.gen_no = gen_no
        self.args = args

    def initialize_population(self):
        for i in range(self.pop_size):
            indi_no = 'indi%02d%02d'%(self.gen_no, i)
            indi = Individual(indi_no, self.args)
            indi.init()
            self.pop.append(indi)

class Individual():
    def __init__(self, indi_no, args):
        self.indi_no = indi_no
        self.obj = -1.0
        self.args = args
        self.loss_weights = []
        self.net_type = []

        self.rnn_layers_num = []
        self.rnn_hidden_size = []
        self.rnn_bidirection = []

        self.cnn_layers_num = []
        self.cnn_kernel_size = []

    def init(self):

        for _ in range(self.args.dec_num + 1):
            self.net_type.append(random.choice(self.args.net))

        for i in range(self.args.dec_num + 2):
            self.loss_weights.append(random.choice(self.args.weights_loss))

        for net in self.net_type:
            if net == 'CNN':
                self.rnn_layers_num.append(None)
                self.rnn_hidden_size.append(None)
                self.rnn_bidirection.append(None)
                self.cnn_kernel_size.append(random.randint(self.args.cnn_kernels_min, self.args.cnn_kernels_max))
                self.cnn_layers_num.append(random.randint(self.args.cnn_layers_min, min(self.args.cnn_layers_max, int(self.args.n_in/self.cnn_kernel_size[-1]))))
            else:
                self.rnn_layers_num.append(random.randint(self.args.rnn_layers_min, self.args.rnn_layers_max))
                self.rnn_hidden_size.append(random.randint(self.args.rnn_hiddens_min, self.args.rnn_hiddens_max))
                self.rnn_bidirection.append(random.choice(self.args.rnn_bidirection))
                self.cnn_layers_num.append(None)
                self.cnn_kernel_size.append(None)
