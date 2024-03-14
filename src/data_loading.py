import os
import pickle
import numpy as np

def labeled_clips(data_types, fs, window_size, overlap):
    for subject in os.listdir('data/WESAD'):
        with open('data/WESAD/' + subject + '/' + subject +'.pkl', 'rb') as file:
            data = pickle.load(file, encoding='latin1')

            for data_type in data_types:
                for i in range(0, len(data['signal']['wrist'][data_type])):
                    data['signal']['wrist'][data_type]
                    data['label']


if __name__ == '__main__':
    data_types =['BVP', 'EDA', 'TEMP']
    fs = [64, 4, 4]
    window_size = [15, 15, 15]
    overlap = [0.5, 0.5, 0.5]

    labeled_clips(data_types, fs, window_size, overlap)
