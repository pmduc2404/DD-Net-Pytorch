import pickle
import os
import csv
import torch
import numpy as np
from sklearn.model_selection import train_test_split

def load_data(ex = "1"):
    # Load Correct Data
    correct_data_link = "./UIPRMD/Correct_data_S"+ex+".csv"
    correct_data_label = "./UIPRMD/Correct_score_S"+ex+".csv"
    incorrect_data_link = "./UIPRMD/Incorrect_data_S"+ex+".csv"
    incorrect_data_label = "./UIPRMD/Incorrect_score_S"+ex+".csv"

    n_joints, n_features = 39, 3
    with open(correct_data_link) as f:
        csv_f = csv.reader(f)
        Correct_X = np.array(list(csv_f), dtype=float)
        
    _, timesteps = Correct_X.shape[0], Correct_X.shape[1]

    correct_input = Correct_X.reshape(-1, timesteps, n_joints, n_features)
    print(correct_input.shape)
    # Load Correct Labels
    with open(correct_data_label) as f:
        csv_f = csv.reader(f)
        Correct_Y = np.array(list(csv_f), dtype=float)
    
    # Load Incorrect Data
    with open(incorrect_data_link) as f:
        csv_f = csv.reader(f)
        Incorrect_X = np.array(list(csv_f), dtype=float)
    
    incorrect_input = Incorrect_X.reshape(-1, timesteps, n_joints, n_features)
    
    # Load Incorrect Labels
    with open(incorrect_data_label) as f:
        csv_f = csv.reader(f)
        Incorrect_Y = np.array(list(csv_f), dtype=float)
    
    data_combined = torch.tensor(np.concatenate((correct_input, incorrect_input), axis=0), dtype=torch.float32)
    labels_combined = torch.tensor(np.concatenate((Correct_Y, Incorrect_Y), axis=0), dtype=torch.float32)

    # Split the data into training and test sets
    train_data, test_data, train_labels, test_labels = train_test_split(data_combined, labels_combined, test_size=0.2, random_state=42)
    # save to pickle files

    train_dataset = {'pose': train_data, 'labels': train_labels}
    test_dataset = {'pose': test_data, 'labels': test_labels}
    
    # create directory for each example
    if not os.path.exists(f'EX_{ex}'):
        os.makedirs(f'EX_{ex}')

    with open(f'EX_{ex}/train_dataset.pkl', 'wb') as f:
        pickle.dump(train_dataset, f)
    with open(f'EX_{ex}/test_dataset.pkl', 'wb') as f:
        pickle.dump(test_dataset, f)

    return train_dataset, test_dataset