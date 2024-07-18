import model_utils
import numpy as np
import pandas as pd
import neo
import matplotlib.pyplot as plt
import scipy.io as sio
from scipy import signal
from scipy.interpolate import interp1d
import spike_utils
import elephant
import quantities as pq
from sklearn.model_selection import ShuffleSplit
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import torch
import torch
from torch import nn
import torch.nn.functional as F
from joblib import Parallel, delayed
import multiprocessing
import Neural_Decoding
import pickle
import seaborn as sns
from functools import partial
from hnn_core.utils import smooth_waveform
import os

sns.set()
sns.set_style("white")

def make_dataframes():
    num_cores = multiprocessing.cpu_count()

    scaler = StandardScaler()

    # CUDA for PyTorch
    use_cuda = torch.cuda.is_available()
    #device = torch.device("cuda:0")
    device = torch.device('cpu')

    torch.backends.cudnn.benchmark = True

    video_paths = [
        '/users/ntolley/scratch/SEE_analysis/experiment_recordings/Anipose-Jun5-2023/pose-3d/Spike_06-05-0818.csv',
        # '/users/ntolley/scratch/SEE_analysis/experiment_recordings/Anipose-Jun6-2023/pose-3d/Spike_06-06-0905.csv',
        '/users/ntolley/scratch/SEE_analysis/experiment_recordings/Anipose-Jun9-2023/pose-3d/Spike_06-09-0907.csv'
    ]

    neural_paths = [
        '/users/ntolley/scratch/SEE_analysis/experiment_recordings/Analysis_Jun05_2023',
        # '/users/ntolley/scratch/SEE_analysis/experiment_recordings/Analysis_June6_2023',
        '/users/ntolley/scratch/SEE_analysis/experiment_recordings/Analysis_Jun09_2023'
    ]

    session_names = [
        'SPKRH20230605',
        # 'SPKRH20230606',
        'SPKRH20230609'
    ]

    cb_names = [
        'Spike_Jun05',
        # 'Spike_Jun06',
        'Spike_Jun09'
    ]


    for file_idx in range(len(video_paths)):
        pos_fname = video_paths[file_idx]
        neural_path = neural_paths[file_idx]
        session_name = session_names[file_idx]
        cb_name = cb_names[file_idx]

        print(f'Making dataframes for {session_name}')


        cam_df = pd.read_csv(pos_fname)

        pos_mask = np.logical_or.reduce((cam_df.columns.str.contains(pat='_x'), cam_df.columns.str.contains(pat='_y'), 
                                    cam_df.columns.str.contains(pat='_z')))
        marker_names = cam_df.columns[pos_mask].values.tolist()

        # Set threshold for likelihood
        score_threshold = 0.5

        # Pull out marker names stripped of suffix (only markers have scores and likelihood DLC variables)
        score_mask = cam_df.columns.str.contains(pat='_x')
        score_names = cam_df.columns[score_mask].values
        marker_names_stripped = cam_df.columns[score_mask].str.split('_').str[:-1].str.join('_').values

        # marker_pos_names indicates which vars are stored in kinematic_df. Only append _x, _y, _z markers
        marker_pos_names = list()
        for mrk_name in marker_names_stripped:
            # Separate likelihood and position data for 
            mrk_pos_mask = np.logical_and(cam_df.columns.str.contains(pat=mrk_name), 
                                            np.logical_or.reduce((cam_df.columns.str.contains(pat='_x'), 
                                                                cam_df.columns.str.contains(pat='_y'),
                                                                cam_df.columns.str.contains(pat='_z')
                                                                )))   
            # There should only be 1 likelihood variable, and 2 or 3 position variables
            #assert np.sum(mrk_pos_mask) == 2 or np.sum(mrk_pos_mask) == 3
            pos_name_list = cam_df.columns[mrk_pos_mask].values.tolist()
            marker_pos_names.extend(pos_name_list)

            # mrk_likelihood = cam_df[f'{mrk_name}_likelihood']
            # for pos_name in pos_name_list:
            #     pos_data = cam_df[pos_name]
                
            #     pos_data[mrk_likelihood < score_threshold] = np.nan

            #     # Update dataframe
            #     cam_df[pos_name] = pos_data

            # Interpolate on NaN values
            null_percent = cam_df.isnull().astype(int).sum().values / len(cam_df)
            #non_null_cols = cam_df.columns[null_percent < 0.9]
            #cam_df[non_null_cols] = cam_df[non_null_cols].interpolate(method='cubic')
        print('Finished processing DLC tracking data')

        cb_dict = dict()
        unit_idx = 0
        stream_indices = [1, 0] 
        for file_idx, cb_idx in enumerate(range(1,3)):
            # Use neo module to load blackrock files
            experiment_dict = sio.loadmat(f'{neural_path}/eventsCB{cb_idx}_{cb_name}.mat')
            nev = neo.io.BlackrockIO(f'{neural_path}/{session_name}_CB{cb_idx}_quiver4toyPKPK4Rotation_delay/derived/{session_name}_CB{cb_idx}_quiver4toyPKPK4Rotation_delay001_RETVR_DSXII.nev')
            ns2 = neo.io.BlackrockIO(f'{neural_path}/{session_name}_CB{cb_idx}_quiver4toyPKPK4Rotation_delay/{session_name}_CB{cb_idx}_quiver4toyPKPK4Rotation_delay001.ns2')

            # Hard-coded values specific to recording equipment
            sampling_rate_list = ns2.header['signal_channels'][['name','sampling_rate']]
            sampling_rate = 30000
            analog_sampling_rate = 1000
            camera_sampling_rate = 40

            # nev seg holds spike train information to extract
            nev_seg = nev.read_segment()
            tstart = nev_seg.t_start.item()
            tstop = nev_seg.t_stop.item()

            # Group spiketrain timestamps by unit id
            unit_timestamps = dict()
            for st in nev_seg.spiketrains:
                if st.annotations['unit_id'] == 1:
                    unit_timestamps[unit_idx] = st.times
                    unit_idx += 1

            # Grab indices for camera frames
            cam_trigger = ns2.get_analogsignal_chunk(channel_names=['FlirCam'], stream_index=stream_indices[file_idx]).transpose()[0]
            num_analog_samples = len(cam_trigger)
            trigger_val = 10000 # threshold where rising edge aligns frame, may need to tweak
            cam_frames = np.flatnonzero((cam_trigger[:-1] < trigger_val) & (cam_trigger[1:] > trigger_val))+1

            cb_dict[f'cb{cb_idx}'] = {'tstart': tstart, 'tstop': tstop, 'unit_timestamps': unit_timestamps,
                                    'cam_frames': cam_frames, 'experiment_dict': experiment_dict}
            
            del nev
            del ns2
            print(f'Finished loading CB{cb_idx} neural data')
        
        experiment_dict = cb_dict['cb1']['experiment_dict']
        cam_frames =  cb_dict['cb1']['cam_frames']


        #Load variables from struct (struct indexing is unfortunately hideous)
        ev_ex = experiment_dict['ans']
        tgtON = ev_ex['tgtON_C'][0][0][0]
        gocON = ev_ex['gocON_C'][0][0][0]
        gocOFF = ev_ex['gocOFF'][0][0][0]
        stmv = ev_ex['stmv_C'][0][0][0]
        contact = ev_ex['contact_C'][0][0][0]
        endhold = ev_ex['endhold_C'][0][0][0]
        layout = ev_ex['LAYOUT_C'][0][0][0]
        position = ev_ex['POSITION_C'][0][0][0]
        reward = ev_ex['reward'][0][0][0]
        error = ev_ex['error'][0][0][0]

        #Define game event for alignment, and window around marker
        event_ts = list(zip(gocON, contact))
        num_events = len(event_ts)

        # Find scale/timeshift between CB1 and CB2
        cb2_align_ts = cb_dict['cb2']['experiment_dict']['ans']['gocON_C'][0][0][0]
        # assert len(cb2_align_ts) == len(gocON)  # **Really important to check why there's a mismatch for June 5 2023**
        cb2_start, cb2_end = cb2_align_ts[0], cb2_align_ts[-1]

        ts_shift = gocON[0] - cb2_start 
        ts_scale = (cb2_end - cb2_start) / (gocON[-1] -  gocON[0])

        unit_timestamps = cb_dict['cb1']['unit_timestamps'].copy()

        # Shift and scale time stamps between the two machines
        unit_timestamps_cb2 = cb_dict['cb2']['unit_timestamps'].copy()
        unit_timestamps_cb2_corrected = dict()
        for unit_idx, unit_ts in unit_timestamps_cb2.items():
            ts_corrected = (unit_ts + ts_shift * (pq.s)) / (ts_scale * pq.s)
            unit_timestamps_cb2_corrected[unit_idx] = ts_corrected


        unit_timestamps.update(unit_timestamps_cb2_corrected)

        print('Calculating single unit firing rates')
        #Append convolved firing rates to dataframe
        kernel_halfwidth = 0.250 #in seconds
        #kernel = elephant.kernels.RectangularKernel(sigma=kernel_halfwidth/np.sqrt(3)*pq.s) 
        kernel = elephant.kernels.ExponentialKernel(sigma=kernel_halfwidth/np.sqrt(3)*pq.s) 
        sampling_period = 0.01*pq.s

        pre_event_time = 0.7 # how much time to store before e_start

        #List to store neural data
        rate_col = list()
        pre_rate_col = list()
        rate_video_col = list()
        unit_col = list()
        trial_col_neural = list()
        layout_col_neural = list()
        position_col_neural = list()

        #List to store kinematic data
        posData_col = list()
        pre_posData_col = list()
        name_col = list()
        trial_col_kinematic = list()
        layout_col_kinematic = list()
        position_col_kinematic = list()

        kinematic_metadata = dict()
        neural_metadata = dict()
        for e_idx, (e_start, e_stop) in enumerate(event_ts):
            e_start = e_start - pre_event_time

            event_length = int((e_stop - (e_start)) / (sampling_period).item())

            print(e_idx, end=' ')
            
            # Load kinematic data
            # Identify which frames fall in the time window
            frame_mask = np.logical_and(cam_frames > (e_start * analog_sampling_rate), cam_frames < (e_stop * analog_sampling_rate))
            frame_idx = np.flatnonzero(frame_mask) #Pull out indeces of valid frames
            frame_times = cam_frames[frame_idx] / analog_sampling_rate

            kinematic_metadata[e_idx] = {'time_data':np.linspace(e_start, e_stop, event_length)}
            
            for mkr in marker_pos_names:
                marker_pos = cam_df[mkr].values[frame_idx]
                f = interp1d(np.linspace(0,1,marker_pos.size), marker_pos)
                marker_interp = f(np.linspace(0,1,event_length ))
                posData = marker_interp[:event_length]
                
                posData_col.append(posData)
                name_col.append(mkr)
                trial_col_kinematic.append(e_idx)
                layout_col_kinematic.append(layout[e_idx])
                position_col_kinematic.append(position[e_idx])
            
            # Load neural data
            for unit_idx, unit_ts in unit_timestamps.items():
                rate = spike_utils.spike_train_rates(unit_ts, e_start, e_stop + 0.1 , sampling_rate, kernel, sampling_period).transpose()

                # Sampling rate to same length of video frames
                f = interp1d(np.linspace(e_start, e_stop, rate.size), rate)
                rate_video = f(frame_times).squeeze()

                # Ensure instantaneous spike train rates match length of interpolated marker trajectory
                rateData = rate[0][:event_length]

                rate_col.append(rateData)
                rate_video_col.append(rate_video)
                
                trial_col_neural.append(e_idx)
                layout_col_neural.append(layout[e_idx])
                position_col_neural.append(position[e_idx])
                unit_col.append(str(unit_idx))
                neural_metadata[e_idx] = {'time_data':frame_idx}

            #-------------------------------
            # Add time data neural
            unit_col.append('time')
            rate_col.append(np.arange(event_length))
            rate_video_col.append(rate_video)
            
            trial_col_neural.append(e_idx)
            layout_col_neural.append(layout[e_idx])
            position_col_neural.append(position[e_idx])

            # Add time data kinematic
            name_col.append('time')
            posData_col.append(np.arange(event_length))
            
            trial_col_kinematic.append(e_idx)
            layout_col_kinematic.append(layout[e_idx])
            position_col_kinematic.append(position[e_idx])
            #-------------------------------

            # One hot encoding of layout information
            for layout_idx in range(1,5):
                if layout_idx == layout[e_idx]:
                    onehot_data = np.ones(event_length)
                else:
                    onehot_data = np.zeros(event_length)

                # Kinematic
                posData_col.append(onehot_data)
                name_col.append(f'layout_{layout_idx}')
                trial_col_kinematic.append(e_idx)
                layout_col_kinematic.append(layout[e_idx])
                position_col_kinematic.append(position[e_idx])

                # Neural
                rate_col.append(onehot_data)
                rate_video_col.append(onehot_data)

                unit_col.append(f'layout_{layout_idx}')
                trial_col_neural.append(e_idx)
                layout_col_neural.append(layout[e_idx])
                position_col_neural.append(position[e_idx])
                neural_metadata[e_idx] = {'time_data':frame_idx}

            # One hot encoding of position information
            for position_idx in range(1,5):
                if position_idx == position[e_idx]:
                    onehot_data = np.ones((event_length,))
                else:
                    onehot_data = np.zeros(event_length)

                # Kinematic
                posData_col.append(onehot_data)
                name_col.append(f'position_{position_idx}')
                trial_col_kinematic.append(e_idx)
                layout_col_kinematic.append(layout[e_idx])
                position_col_kinematic.append(position[e_idx])

                # Neural
                rate_col.append(onehot_data)
                rate_video_col.append(onehot_data)
                unit_col.append(f'position_{position_idx}')
                trial_col_neural.append(e_idx)
                layout_col_neural.append(layout[e_idx])
                position_col_neural.append(position[e_idx])
                neural_metadata[e_idx] = {'time_data':frame_idx}
                
        #Pickle convolved rates
        neural_dict = {'rates':rate_col,  'rates_video': rate_video_col,'unit':unit_col,
                    'trial':trial_col_neural, 'layout': layout_col_neural, 'position': position_col_neural}
        neural_df = pd.DataFrame(neural_dict)
        neural_df['count'] = neural_df['rates'].apply(np.sum)

        # Pickle kinematic tracking
        kinematic_dict = {'name':name_col, 'posData':posData_col,
                        'trial':trial_col_kinematic, 'layout': layout_col_kinematic, 'position': position_col_kinematic}
        kinematic_df = pd.DataFrame(kinematic_dict)

        metadata={'kinematic_metadata':kinematic_metadata, 'neural_metadata':neural_metadata, 'num_trials':num_events, 'kernel_halfwidth':kernel_halfwidth}

        save_path = f'/users/ntolley/scratch/SEE_analysis/processed_data/{session_name}'
        os.makedirs(save_path, exist_ok=True)

        #Save DataFrames to temporary folder
        kinematic_df.to_pickle(f'{save_path}/kinematic_df.pkl')
        neural_df.to_pickle(f'{save_path}/neural_df.pkl')

        #Save metadata
        output = open(f'{save_path}/metadata.pkl', 'wb')
        pickle.dump(metadata, output)
        output.close()

        output = open(f'{save_path}/cb_dict.pkl', 'wb')
        pickle.dump(cb_dict, output)
        output.close()


        

if __name__ == '__main__':
    make_dataframes()
