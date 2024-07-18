import sys
sys.path.append('../code')
# sys.path.append('../externals/mamba/')

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
from sklearn.model_selection import ShuffleSplit
import pickle
import model_utils
from torch import nn
from mamba_ssm import Mamba
import Neural_Decoding


# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0")
# device = torch.device('cpu')

torch.backends.cudnn.benchmark = True

#LSTM/GRU architecture for decoding
#RNN architecture for decoding kinematics
class model_mamba(nn.Module):
    def __init__(self, input_size, output_size, d_model, d_state=16, d_conv=4, expand=2, dropout=0.2, device=device,
                 cat_features=None, task_info=True):
        super(model_mamba, self).__init__()

        # Defining some parameters
        self.device = device
        self.dropout = dropout
        self.cat_features = cat_features
        self.task_info = task_info

        self.input_size = input_size

        if self.cat_features is not None:
            self.num_cat_features = np.sum(self.cat_features).astype(int)
            self.input_size = self.input_size - self.num_cat_features

            
        # self.fc = nn.Linear(in_features=d_model, out_features=output_size).to(device)
        self.fc1 = model_utils.model_ann(d_model, output_size, [100, 100]).to(device)
        self.fc2 = nn.Linear(in_features=output_size, out_features=output_size).to(device)
        self.dropout = nn.Dropout(p=self.dropout)

        # self.fc = nn.Linear((input_), output_size)
        self.mamba = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand).to(device)
    
    def forward(self, x):

        if not self.task_info and self.cat_features is not None:
            x[:, :, self.cat_features] = 0.0
            x[:,:,-np.random.choice([0,1,2,3])] = 1.0

        x[:,:,-50:] # randomly drop to 50 neurons
        batch_size = x.size(0)

        out = self.mamba(x)
        out = self.dropout(out)
        out = self.fc1(out)
        out = self.fc2(out)
        return out, None, None

def run_rnn(pred_df, neural_df, neural_offset, cv_dict, metadata, task_info=True,
            window_size=50, num_cat=0, label_col=None, flip_outputs=False, dropout=0.5):
    exclude_processing = None
    criterion = model_utils.mse
    if num_cat > 0:
        exclude_processing = np.zeros(len(neural_df['unit'].unique()))
        exclude_processing[-num_cat:] = np.ones(num_cat)
        exclude_processing = exclude_processing.astype(bool)

    # else:
    #     criterion = mse

    data_arrays, generators = model_utils.make_generators(
    pred_df, neural_df, neural_offset, cv_dict, metadata, exclude_neural=exclude_processing,
    window_size=window_size, flip_outputs=flip_outputs, batch_size=1000, label_col=label_col, fold=0, data_step_size=10)

    # Unpack tuple into variables
    training_set, validation_set, testing_set = data_arrays
    training_generator, training_eval_generator, validation_generator, testing_generator = generators

    X_train_data = training_set[:][0][:,-1,:].detach().cpu().numpy()
    y_train_data = training_set[:][1][:,-1,:].detach().cpu().numpy()

    X_test_data = testing_set[:][0][:,-1,:].detach().cpu().numpy()
    y_test_data = testing_set[:][1][:,-1,:].detach().cpu().numpy()

    #Define hyperparameters
    # lr = 1e-2
    lr = 1e-3
    # weight_decay = 1e-4
    weight_decay = 0.0
    max_epochs = 1000
    input_size = X_train_data.shape[1] 
    output_size = y_train_data.shape[1] 

    model_rnn = model_mamba(input_size, output_size, d_model=input_size,
                            d_state=128, d_conv=4, expand=2, cat_features=exclude_processing,
                            task_info=task_info)


    # Define Loss, Optimizerints h
    optimizer = torch.optim.Adam(model_rnn.parameters(), lr=lr, weight_decay=weight_decay)

    #Train model
    loss_dict = model_utils.train_validate_model(model_rnn, optimizer, criterion, max_epochs, training_generator, validation_generator, device, 10, 10)

    #Evaluate trained model
    rnn_train_pred = model_utils.evaluate_model(model_rnn, training_eval_generator, device)
    rnn_test_pred = model_utils.evaluate_model(model_rnn, testing_generator, device)

    rnn_train_corr = model_utils.matrix_corr(rnn_train_pred, y_train_data)
    rnn_test_corr = model_utils.matrix_corr(rnn_test_pred, y_test_data)

    res_dict = {'loss_dict': loss_dict,
                'train_pred': rnn_train_pred, 'test_pred': rnn_test_pred,
                'train_corr': rnn_train_corr, 'test_corr': rnn_test_corr}


    
if __name__ == '__main__':

    session_names = [
        'SPKRH20230605',
        # 'SPKRH20230606',
        'SPKRH20230609'
    ]

    for session_name in session_names:
        noise_fold = 0
        data_dict = model_utils.get_marker_decode_dataframes(noise_fold=noise_fold)
        wrist_df = data_dict['wrist_df']
        task_neural_df = data_dict['task_neural_df']
        notask_neural_df = data_dict['notask_neural_df']
        metadata = data_dict['metadata']
        cv_dict = data_dict['cv_dict']

        neuron_list = notask_neural_df['unit'].unique()

        notask_time_neural_mask = notask_neural_df['unit'] != 'time'
        notask_neural_df = notask_neural_df[notask_time_neural_mask]

        task_time_neural_mask = task_neural_df['unit'] != 'time'
        task_neural_df = task_neural_df[task_time_neural_mask]

        wrist_mask = wrist_df['name'] != 'time'
        wrist_df = wrist_df[wrist_mask]

        # func_dict = {'rnn': run_rnn, 'wiener': run_wiener}
        func_dict = {'rnn': run_rnn}

        fpath = '../data/neuron_num_results/'


        # df_dict = {'task': {'df': task_neural_df, 'task_info': True, 'num_cat': 4, 'flip_outputs': True},
        #             'notask': {'df': notask_neural_df, 'task_info': False, 'num_cat': 0, 'flip_outputs': True}}

        df_dict = {'task': {'df': task_neural_df, 'task_info': True, 'num_cat': 4, 'flip_outputs': True},
                    'notask': {'df': task_neural_df, 'task_info': False, 'num_cat': 4, 'flip_outputs': True}}

                
        decode_results = dict()
        for func_name, func in func_dict.items():
            decode_results[func_name] = dict()
            for df_type, pred_df in df_dict.items():
                # print(f'{func_name}_{df_type} num_neurons: {num_neurons}; repeat {repeat_idx}')

                model, res_dict = func(wrist_df, pred_df['df'], neural_offset, cv_dict, metadata, task_info=pred_df['task_info'],
                                        window_size=window_size, num_cat=pred_df['num_cat'], label_col=label_col, flip_outputs=pred_df['flip_outputs'])

                decode_results[func_name][df_type] = res_dict

                # # Save results on every loop in case early stop
                # num_neuron_results_dict[f'repeat_{repeat_idx}'][f'num_neuron_{num_neurons}'] = decode_results
                # # #Save metadata
                # output = open(f'{fpath}num_neuron_results.pkl', 'wb')
                # pickle.dump(num_neuron_results_dict, output)
                # output.close()

                # if func_name == 'rnn':
                #     torch.save(model.state_dict(), f'{fpath}models/{df_type}_neurons{num_neurons}_repeat{repeat_idx}.pt')



