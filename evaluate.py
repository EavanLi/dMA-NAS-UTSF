import importlib

class FitnessEvaluate(object):
    def __init__(self, pop, args):
        self.pop = pop
        self.n_in = args.n_in
        self.args = args

    def generate_to_python_file(self):
        for indi in self.pop:
            indi_no = indi.indi_no
            dec_num = indi.args.dec_num
            net_type = indi.net_type

            rnn_bidirection = indi.rnn_bidirection
            rnn_layers_num = indi.rnn_layers_num
            rnn_hidden_size = indi.rnn_hidden_size

            cnn_kernel_size = indi.cnn_kernel_size
            cnn_layers_num = indi.cnn_layers_num

            init_model_statement = []

            # ------------------------------------__init__ self.op--------------------------------------------------
            for i in range(dec_num):
                if net_type[i] == 'CNN':
                    for layer_No in range(cnn_layers_num[i]):
                        statement = 'self.op%d_%d = nn.Conv1d(1, 1, ' % (i, layer_No) + str(cnn_kernel_size[i]) + ')'
                        init_model_statement.append(statement)
                    init_model_statement.append('self.maxpool%d = nn.MaxPool1d(2, 2)' % (i))
                else:
                    if rnn_bidirection[i]:
                        statement = 'self.op%d = nn.%s(input_size=1, hidden_size=%d, num_layers=%d, batch_first=True, bidirectional=True)'%(i, net_type[i], rnn_hidden_size[i], rnn_layers_num[i])
                    else:
                        statement = 'self.op%d = nn.%s(input_size=1, hidden_size=%d, num_layers=%d, batch_first=True, bidirectional=False)' % (i, net_type[i], rnn_hidden_size[i], rnn_layers_num[i])
                    init_model_statement.append(statement)

            # ------------------------------------__init__ self.op_fc-----------------------------------------------
            for i in range(dec_num):
                if net_type[i] == 'CNN':

                    l_in = self.n_in
                    for _ in range(cnn_layers_num[i]):
                        l_in = l_in - cnn_kernel_size[i] + 1
                    l_in = int(max(l_in/2, 1))
                    statement = 'self.op_fc%d = nn.Linear(%d, n_out)' % (i, l_in)
                    init_model_statement.append(statement)
                else:
                    if rnn_bidirection[i]:
                        statement = 'self.op_fc%d = nn.Linear(%d*2, n_out)'%(i, rnn_hidden_size[i])
                    else:
                        statement = 'self.op_fc%d = nn.Linear(%d, n_out)' % (i, rnn_hidden_size[i])
                    init_model_statement.append(statement)

            # ------------------------------------__init__ self.op_org----------------------------------------------
            if net_type[-1] == 'CNN':
                for layer_No in range(cnn_layers_num[-1]):
                    statement = 'self.op_org_%d = nn.Conv1d(1, 1, ' % (layer_No) + str(cnn_kernel_size[-1]) + ')'
                    init_model_statement.append(statement)
                init_model_statement.append('self.maxpool_org = nn.MaxPool1d(2, 2)')
            else:
                init_model_statement.append('self.op_org = nn.%s(input_size=1, hidden_size=%d, num_layers=%d, batch_first=True, bidirectional=%s)' % (net_type[-1], rnn_hidden_size[-1], rnn_layers_num[-1], rnn_bidirection[-1]))
            # ------------------------------------__init__ self.op_fc_org------------------------------------------
            if net_type[-1] == 'CNN':
                l_in = self.n_in
                for _ in range(cnn_layers_num[-1]):
                    l_in = l_in - cnn_kernel_size[-1] + 1
                l_in = int(max(l_in / 2, 1))
                init_model_statement.append('self.op_fc_org = nn.Linear(%d, n_out)' % (l_in))
            else:
                if rnn_bidirection[-1]:
                    init_model_statement.append('self.op_fc_org = nn.Linear(' + str(rnn_hidden_size[-1] * 2) + ', n_out)')
                else:
                    init_model_statement.append('self.op_fc_org = nn.Linear(' + str(rnn_hidden_size[-1]) + ', n_out)')


            forward_statement = [] # forward_statement.append()
            forward_statement.append('x.requires_grad_()')
            forward_statement.append('for dec_No in range(self.dec_num):')
            dec_No = 0

            # --------------------------------------forward-------------------------------------------
            while dec_No < dec_num:
                if dec_No == 0:
                    forward_statement.append('    if dec_No == 0:')
                else:
                    forward_statement.append('    elif dec_No == %d:' % (dec_No))

                if net_type[dec_No] == 'CNN':
                    forward_statement.append('        output%d = x[dec_No].to(device="cuda").view(len(x[dec_No]), 1, -1)' % (dec_No))
                    for layer_No in range(cnn_layers_num[dec_No]):
                        forward_statement.append('        output%d = torch.relu(self.op%d_%d(output%d))' % (dec_No, dec_No, layer_No, dec_No))
                    forward_statement.append('        output%d = self.maxpool%d(output%d).squeeze(1)' % (dec_No, dec_No, dec_No))
                    forward_statement.append('        output%d = self.op_fc%d(output%d)' % (dec_No, dec_No, dec_No))
                    dec_No += 1
                else:
                    if net_type[dec_No] in ['GRU','RNN']:
                        forward_statement.append('        _, h%d = self.op%d(x[dec_No].to(device="cuda").view(len(x[dec_No]), -1, 1))' % (dec_No, dec_No))
                    elif net_type[dec_No] in ['LSTM']:
                        forward_statement.append('        _, (h%d, _) = self.op%d(x[dec_No].to(device="cuda").view(len(x[dec_No]), -1, 1))' % (dec_No, dec_No))
                    if rnn_bidirection[dec_No]:
                        forward_statement.append('        h%d = h%d.view(' % (dec_No, dec_No) + str(rnn_layers_num[dec_No]) + ', 2, x[dec_No].shape[0], ' + str(rnn_hidden_size[dec_No]) + ')')
                        forward_statement.append('        f_hidden, b_hidden = h%d[-1]' % (dec_No))
                        forward_statement.append('        h%d = torch.cat((f_hidden, b_hidden), dim=1)' % (dec_No))
                    else:
                        forward_statement.append('        h%d = h%d.view(' % (dec_No, dec_No) + str(rnn_layers_num[dec_No]) + ', 1, x[dec_No].shape[0], ' + str(rnn_hidden_size[dec_No]) + ')')
                        forward_statement.append('        h%d = h%d[-1].squeeze(0)' % (dec_No, dec_No))
                    forward_statement.append('        output%d = self.op_fc%d(h%d)' % (dec_No, dec_No, dec_No))
                    dec_No += 1

            dec_No, cat_statement = 0, ''
            while dec_No < dec_num:
                cat_statement = cat_statement + 'output%d' % (dec_No) + ', '
                dec_No += 1
            forward_statement.append('output_dec = torch.cat(['+cat_statement[:-2] + '], dim=0)')


            forward_statement.append('# -------------------------------------------直接对train_org_x预测------------------------------------------------')
            if net_type[-1] == 'CNN':
                forward_statement.append('output_org = x[-1].to(device="cuda").view(len(x[-1]), 1, -1)')
                for layer_No in range(cnn_layers_num[dec_No]):
                    forward_statement.append('output_org = torch.relu(self.op_org_%d(output_org))' % (layer_No))
                forward_statement.append('output_org = self.maxpool_org(output_org).squeeze(1)')
                forward_statement.append('output_org = self.op_fc_org(output_org)')
            else:
                if net_type[-1] in ['GRU', 'RNN']:
                    forward_statement.append('_, h_org = self.op_org(x[-1].to(device="cuda").view(len(x[-1]), -1, 1))')
                elif net_type[-1] in ['LSTM']:
                    forward_statement.append('_, (h_org, _) = self.op_org(x[-1].to(device="cuda").view(len(x[-1]), -1, 1))')
                else:
                    raise ValueError("The net type is not defined.")
                if rnn_bidirection[-1]:
                    forward_statement.append('h_org = h_org.view(' + str(rnn_layers_num[-1]) + ', 2, x[dec_No].shape[0], ' + str(rnn_hidden_size[-1]) + ')')
                    forward_statement.append('f_hidden, b_hidden = h_org[-1]')
                    forward_statement.append('h_org = torch.cat((f_hidden, b_hidden), dim=1)')
                else:
                    forward_statement.append('h_org = h_org.view(' + str(rnn_layers_num[-1]) + ', 1, x[dec_No].shape[0], ' + str(rnn_hidden_size[-1]) + ')')
                    forward_statement.append('h_org = h_org[-1].squeeze(0)')
                forward_statement.append('output_org = self.op_fc_org(h_org)')

            forward_statement.append('# -------------------------------------------将output_dec和output_org拼接----------------------------------------')
            forward_statement.append('output_com = torch.stack([' + cat_statement[:-2] + ', output_org], dim = 2)')
            forward_statement.append('output_com = self.dense1(output_com).squeeze(2)')
            forward_statement.append('output = torch.cat([output_dec, output_org, output_com], dim=0)')

            part1, part2, part3 = self.read_template()
            _str = []
            _str.extend(part1)
            for s in init_model_statement:
                _str.append('        %s' % (s))
            _str.extend(part2)
            for s in forward_statement:
                _str.append('        %s' % (s))
            _str.extend(part3)

            file_name = './%s/scripts/%s.py' % (self.args.time_series_name, indi_no) #file_name = './scripts/%s.py' % (indi_no)
            script_file_handler = open(file_name, 'w')
            script_file_handler.write('\n'.join(_str))
            script_file_handler.flush()
            script_file_handler.close()

    def read_template(self):
        _path = 'template_model.py'
        part1 = []
        part2 = []
        part3 = []

        f = open(_path, encoding='utf-8')
        f.readline() #skip this comment
        line = f.readline().rstrip()
        while line.strip() != '#generated_init':
            part1.append(line)
            line = f.readline().rstrip()

        line = f.readline().rstrip() #skip the comment '#generated_init'
        while line.strip() != '#generated_forward':
            part2.append(line)
            line = f.readline().rstrip()

        line = f.readline().rstrip() #skip the comment '#generate_forward'
        while line.strip() != '"""':
            part3.append(line)
            line = f.readline().rstrip()

        return part1, part2, part3

    def evaluate(self, args, train_dec_org_x, train_dec_org_y, valid_dec_x, valid_dec_y, valid_org_x, valid_org_y, test_dec_x, test_dec_y, test_org_x, test_org_y):
        for indi in self.pop:
            _module = importlib.import_module('.', '%s.scripts.%s' % (self.args.time_series_name, indi.indi_no))
            cls_obj = getattr(_module, 'RunModel')(args, indi, train_dec_org_x, train_dec_org_y, valid_dec_x, valid_dec_y, valid_org_x, valid_org_y, test_dec_x, test_dec_y, test_org_x, test_org_y)  # 进入scripts中每个个体的文件中进行训练
            try:
                indi_obj = cls_obj.process()
                indi.obj = indi_obj
            except:
                print(self.args.time_series_name + ', ' + indi.indi_no + ': Out of memory!')