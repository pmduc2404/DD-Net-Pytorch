#! /usr/bin/env python
#! coding:utf-8
from pathlib import Path
import matplotlib.pyplot as plt
from torch import log
from tqdm import tqdm
import torch
import torch.nn as nn
import argparse
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import confusion_matrix

# from dataloader.jhmdb_loader import load_jhmdb_data, Jdata_generator, JConfig
# from dataloader.shrec_loader import load_shrec_data, Sdata_generator, SConfig
from dataloader.uiprmd_loader import load_UIPRMD_data, UIPRMDdata_generator, UIPRMDConfig

from models.DDNet_Original import DDNet_Original as DDNet
# from models.DDNet_Original_ODE import DDNet_GDE as DDNet

from utils import makedir
import sys
import time
import numpy as np
import logging
sys.path.insert(0, './pytorch-summary/torchsummary/')
from torchsummary import summary  # noqa

savedir = Path('experiments') / Path(str(int(time.time())))
makedir(savedir)
logging.basicConfig(filename=savedir/'train.log', level=logging.INFO)
history = {
    "train_loss": [],
    "test_loss": [],
    "test_mad": [],
    "test_mape": [],
    "test_rmse": []
}

edge_index = torch.tensor([
    [0, 0, 0, 0, 3, 4, 7, 7, 27, 29, 31, 33, 28, 30, 32, 34, 7, 10, 10, 13, 15, 17, 19, 21, 23, 10, 14, 16, 18, 20, 22, 24],  
    [1, 2, 3, 4, 7, 7, 27, 28, 29, 31, 33, 37, 30, 32, 34, 38, 10, 13, 14, 15, 17, 19, 21, 23, 25, 14, 16, 18, 20, 22, 24, 26]   
    ])

def compute_metrics(y_true, y_pred):
    abs_error = torch.abs(y_true - y_pred)
    square_error = (y_true - y_pred) ** 2

    mad = abs_error.mean().item()  
    mape = (abs_error / (y_true + 1e-8)).mean().item() * 100 
    rmse = torch.sqrt(square_error.mean()).item()  

    return mad, mape, rmse


def train(args, model, device, train_loader, optimizer, epoch, criterion):
    model.train()
    train_loss = 0
    for batch_idx, (data1, data2, target) in enumerate(tqdm(train_loader)):
        M, P, target = data1.to(device), data2.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(M, P)
        loss = criterion(output, target)
        train_loss += loss.detach().item()
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            msg = ('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data1), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            print(msg)
            logging.info(msg)
            if args.dry_run:
                break
    history['train_loss'].append(train_loss)
    return train_loss


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    all_targets = []
    all_predictions = []
    criterion = nn.L1Loss(reduction='sum')  # MAE loss
    
    with torch.no_grad():
        for _, (data1, data2, target) in enumerate(tqdm(test_loader)):
            M, P, target = data1.to(device), data2.to(device), target.to(device)
            output = model(M, P)
            # sum up batch loss
            test_loss += criterion(output, target).item()
            
            # Store predictions and targets for metric calculation
            all_targets.append(target)
            all_predictions.append(output)

    # Concatenate all batches
    all_targets = torch.cat(all_targets, dim=0)
    all_predictions = torch.cat(all_predictions, dim=0)
    
    # Calculate metrics on GPU
    mad, mape, rmse = compute_metrics(all_targets, all_predictions)
    
    # Move tensors to CPU for logging
    all_targets = all_targets.cpu()
    all_predictions = all_predictions.cpu()
    
    test_loss /= len(test_loader.dataset)
    history['test_loss'].append(test_loss)
    history['test_mad'].append(mad)
    history['test_mape'].append(mape)
    history['test_rmse'].append(rmse)
    
    msg = ('Test set: Average loss: {:.4f}, MAD: {:.4f}, MAPE: {:.4f}%, RMSE: {:.4f}\n'.format(
        test_loss, mad, mape, rmse))
    print(msg)
    logging.info(msg)


def main():
    # Training settings
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=199, metavar='N',
                        help='number of epochs to train (default: 199)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001)')
    parser.add_argument('--gamma', type=float, default=0.5, metavar='M',
                        help='Learning rate step gamma (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--log-interval', type=int, default=2, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    parser.add_argument('--dataset', type=int, required=True, metavar='N',
                        help='0 for UIPRMD')
    parser.add_argument('--model', action='store_true', default=False,
                        help='For Saving the current Model')
    parser.add_argument('--calc_time', action='store_true', default=False,
                        help='calc calc time per sample')
    args = parser.parse_args()
    logging.info(args)
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"Using device: {device}")
    kwargs = {'batch_size': args.batch_size}
    if use_cuda:
        kwargs.update({'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True},)

    # alias
    Config = None
    data_generator = None
    load_data = None
    clc_num = 0
    if args.dataset == 0:
        Config = UIPRMDConfig()
        data_generator = UIPRMDdata_generator
        load_data = load_UIPRMD_data
        # clc_num = Config.clc_num
    # elif args.dataset == 1:
    #     Config = SConfig()
    #     load_data = load_shrec_data
    #     clc_num = Config.class_coarse_num
    #     data_generator = Sdata_generator('coarse_label')
    # elif args.dataset == 2:
    #     Config = SConfig()
    #     clc_num = Config.class_fine_num
    #     load_data = load_shrec_data
    #     data_generator = Sdata_generator('fine_label')
    # else:
    #     print("Unsupported dataset!")
    #     sys.exit(1)

    C = Config
    Train, Test = load_data()
    X_0, X_1, Y = data_generator(Train, C)
    X_0 = torch.from_numpy(X_0).type('torch.FloatTensor')
    X_1 = torch.from_numpy(X_1).type('torch.FloatTensor')
    Y = torch.from_numpy(Y).type('torch.FloatTensor')

    X_0_t, X_1_t, Y_t = data_generator(Test, C)
    X_0_t = torch.from_numpy(X_0_t).type('torch.FloatTensor')
    X_1_t = torch.from_numpy(X_1_t).type('torch.FloatTensor')
    Y_t = torch.from_numpy(Y_t).type('torch.FloatTensor')

    trainset = torch.utils.data.TensorDataset(X_0, X_1, Y)
    train_loader = torch.utils.data.DataLoader(trainset, **kwargs)

    testset = torch.utils.data.TensorDataset(X_0_t, X_1_t, Y_t)
    test_loader = torch.utils.data.DataLoader(
        testset, batch_size=args.test_batch_size)

    Net = DDNet(C.frame_l, C.joint_n, C.joint_d,
                C.feat_d, C.filters, clc_num)
    model = Net.to(device)

    summary(model, [(C.frame_l, C.feat_d), (C.frame_l, C.joint_n, C.joint_d)])
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))

    criterion = nn.L1Loss()  # MAE loss
    scheduler = ReduceLROnPlateau(
        optimizer, factor=args.gamma, patience=5, cooldown=0.5, min_lr=5e-6)
    for epoch in range(1, args.epochs + 1):
        train_loss = train(args, model, device, train_loader,
                           optimizer, epoch, criterion)
        test(model, device, test_loader)
        scheduler.step(train_loss)

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=4, ncols=1)
    ax1.plot(history['train_loss'])
    ax1.plot(history['test_loss'])
    ax1.legend(['Train', 'Test'], loc='upper left')
    ax1.set_xlabel('Epoch')
    ax1.set_title('Loss')

    ax2.set_title('MAD')
    ax2.set_ylabel('MAD')
    ax2.set_xlabel('Epoch')
    ax2.plot(history['test_mad'])
    xmin = np.argmin(history['test_mad'])
    ymin = np.min(history['test_mad'])
    text = "x={}, y={:.3f}".format(xmin, ymin)
    ax2.annotate(text, xy=(xmin, ymin))

    ax3.set_title('MAPE')
    ax3.set_ylabel('MAPE (%)')
    ax3.set_xlabel('Epoch')
    ax3.plot(history['test_mape'])
    xmin = np.argmin(history['test_mape'])
    ymin = np.min(history['test_mape'])
    text = "x={}, y={:.3f}".format(xmin, ymin)
    ax3.annotate(text, xy=(xmin, ymin))

    ax4.set_title('RMSE')
    ax4.set_ylabel('RMSE')
    ax4.set_xlabel('Epoch')
    ax4.plot(history['test_rmse'])
    xmin = np.argmin(history['test_rmse'])
    ymin = np.min(history['test_rmse'])
    text = "x={}, y={:.3f}".format(xmin, ymin)
    ax4.annotate(text, xy=(xmin, ymin))

    fig.tight_layout()
    fig.savefig(str(savedir / "perf.png"))
    if args.save_model:
        torch.save(model.state_dict(), str(savedir/"model.pt"))
    if args.calc_time:
        device = ['cpu', 'cuda']
        # calc time
        for d in device:
            tmp_X_0_t = X_0_t.to(d)
            tmp_X_1_t = X_1_t.to(d)
            model = model.to(d)
            # warm up
            _ = model(tmp_X_0_t, tmp_X_1_t)

            tmp_X_0_t = tmp_X_0_t.unsqueeze(1)
            tmp_X_1_t = tmp_X_1_t.unsqueeze(1)
            start = time.perf_counter_ns()
            for i in range(tmp_X_0_t.shape[0]):
                _ = model(tmp_X_0_t[i, :, :, :], tmp_X_1_t[i, :, :, :])
            end = time.perf_counter_ns()
            msg = ("total {}ns, {:.2f}ns per one on {}".format((end - start),
                                                               ((end - start) / (X_0_t.shape[0])), d))
            print(msg)
            logging.info(msg)


if __name__ == '__main__':
    main()
