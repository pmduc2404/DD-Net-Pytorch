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

from dataloader.jhmdb_loader import load_jhmdb_data, Jdata_generator, JConfig
from dataloader.shrec_loader import load_shrec_data, Sdata_generator, SConfig
# from models.DDNet_Original import DDNet_Original as DDNet
from models.DDNet_Original_ODE import DDNet_Original as DDNet
# from models.DDNet_Enhanced_ODE import EnhancedDDNet_ODE as DDNet
from models.OTLoss import *

from utils import makedir
import sys
import time
import numpy as np
import logging

sys.path.insert(0, './pytorch-summary/torchsummary/')
# from torchsummary import summary  # noqa

# Add seed setting for reproducibility
def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Set deterministic behavior for CUDA
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    np.random.seed(seed)

# savedir = Path('experiments') / Path(str(int(time.time())))

def get_incremental_dir(root_dir, prefix="exp"):
    root_dir = Path(root_dir)
    root_dir.mkdir(parents=True, exist_ok=True)
    i = 0
    while True:
        exp_dir = root_dir / (prefix if i == 0 else f"{prefix}{i}")
        if not exp_dir.exists():
            exp_dir.mkdir(parents=True)
            return exp_dir
        i += 1


history = {
    "train_loss": [],
    "test_loss": [],
    "test_acc": []
}

# Set default dtype and device (modern approach)
torch.set_default_dtype(torch.float32)
if torch.cuda.is_available():
    torch.set_default_device('cuda')
else:
    torch.set_default_device('cpu')

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
    correct = 0
    # criterion = nn.CrossEntropyLoss(reduction='sum')
    criterion = create_robust_ot_loss()
    with torch.no_grad():
        for _, (data1, data2, target) in enumerate(tqdm(test_loader)):
            M, P, target = data1.to(device), data2.to(device), target.to(device)
            output = model(M, P)
            # sum up batch loss
            test_loss += criterion(output, target).item()
            # get the index of the max log-probability
            pred = output.argmax(dim=1, keepdim=True)
            # output shape (B,Class)
            # target_shape (B)
            # pred shape (B,1)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    history['test_loss'].append(test_loss)
    history['test_acc'].append(correct / len(test_loader.dataset))
    msg = ('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    print(msg)
    logging.info(msg)


def main():
    # Training settings
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=600, metavar='N',
                        help='number of epochs to train (default: 199)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.01)')
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
    parser.add_argument('--seed', type=int, default=42, metavar='S',
                        help='random seed (default: 42)')
    parser.add_argument('--dataset', type=int, required=True, metavar='N',
                        help='0 for JHMDB, 1 for SHREC coarse, 2 for SHREC fine, others is undefined')
    parser.add_argument('--model', action='store_true', default=False,
                        help='For Saving the current Model')
    parser.add_argument('--calc_time', action='store_true', default=False,
                        help='calc calc time per sample')
    args = parser.parse_args()
    logging.info(args)
    
    # Set random seed before creating data loaders
    set_seed(args.seed)
    
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    
    # Create a generator for DataLoader
    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed)
    
    kwargs = {
        'batch_size': args.batch_size,
        'generator': generator
    }
    
    if use_cuda:
        kwargs.update({
            'num_workers': 1,
            'pin_memory': True,
            'shuffle': True
        })
    
    savedir = get_incremental_dir("experiments", prefix="exp")

    makedir(savedir)
    logging.basicConfig(filename=savedir/'train.log', level=logging.INFO)

    # alias
    Config = None
    data_generator = None
    load_data = None
    clc_num = 0
    if args.dataset == 0:
        Config = JConfig()
        data_generator = Jdata_generator
        load_data = load_jhmdb_data
        clc_num = Config.clc_num
    elif args.dataset == 1:
        Config = SConfig()
        load_data = load_shrec_data
        clc_num = Config.class_coarse_num
        data_generator = Sdata_generator('coarse_label')
    elif args.dataset == 2:
        Config = SConfig()
        clc_num = Config.class_fine_num
        load_data = load_shrec_data
        data_generator = Sdata_generator('fine_label')
    else:
        print("Unsupported dataset!")
        sys.exit(1)

    C = Config
    Train, Test, le = load_data()
    X_0, X_1, Y = data_generator(Train, C, le)
    X_0 = torch.from_numpy(X_0).to(torch.float32)
    X_1 = torch.from_numpy(X_1).to(torch.float32)
    Y = torch.from_numpy(Y).to(torch.long)

    X_0_t, X_1_t, Y_t = data_generator(Test, C, le)
    X_0_t = torch.from_numpy(X_0_t).to(torch.float32)
    X_1_t = torch.from_numpy(X_1_t).to(torch.float32)
    Y_t = torch.from_numpy(Y_t).to(torch.long)

    trainset = torch.utils.data.TensorDataset(X_0, X_1, Y)
    train_loader = torch.utils.data.DataLoader(trainset, **kwargs)

    testset = torch.utils.data.TensorDataset(X_0_t, X_1_t, Y_t)
    test_loader = torch.utils.data.DataLoader(
        testset, batch_size=args.test_batch_size)

    Net = DDNet(C.frame_l, C.joint_n, C.joint_d,
                C.feat_d, C.filters, clc_num)
    model = Net.to(device)
    from torchinfo import summary
    x1 = torch.randn(1, C.frame_l, C.feat_d).to(device)  # batch size 1
    x2 = torch.randn(1, C.frame_l, C.joint_n, C.joint_d).to(device)
    # summary(model, input_data=(x1, x2))

    # summary(model, [(C.frame_l, C.feat_d), (C.frame_l, C.joint_n, C.joint_d)])
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))

    # criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    criterion = create_robust_ot_loss()

    scheduler = ReduceLROnPlateau(
        optimizer, factor=args.gamma, patience=5, cooldown=0.5, min_lr=5e-6)
    
    best_acc = 0.0  # Track best accuracy
    best_model_path = savedir / "model_best.pt"
    last_model_path = savedir / "model_last.pt"

    for epoch in range(1, args.epochs + 1):
        train_loss = train(args, model, device, train_loader,
                           optimizer, epoch, criterion)
        test(model, device, test_loader)
        scheduler.step(train_loss)
        
        # Save best model if current accuracy is better
        current_acc = history['test_acc'][-1]
        if current_acc > best_acc:
            best_acc = current_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'accuracy': best_acc,
            }, str(best_model_path))
            msg = f'Saved new best model with accuracy: {best_acc:.4f}'
            print(msg)
            logging.info(msg)

    fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1)
    ax1.plot(history['train_loss'])
    ax1.plot(history['test_loss'])
    ax1.legend(['Train', 'Test'], loc='upper left')
    ax1.set_xlabel('Epoch')
    ax1.set_title('Loss')

    ax2.set_title('Model accuracy')
    ax2.set_ylabel('Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.plot(history['test_acc'])
    xmax = np.argmax(history['test_acc'])
    ymax = np.max(history['test_acc'])
    text = "x={}, y={:.3f}".format(xmax, ymax)
    ax2.annotate(text, xy=(xmax, ymax))

    ax3.set_title('Confusion matrix')
    model.eval()
    with torch.no_grad():
        Y_pred = model(X_0_t.to(device), X_1_t.to(
            device)).cpu().numpy()
    Y_test = Y_t.numpy()
    cnf_matrix = confusion_matrix(
        Y_test, np.argmax(Y_pred, axis=1))
    ax3.imshow(cnf_matrix)
    fig.tight_layout()
    fig.savefig(str(savedir / "perf.png"))
    if args.save_model:
        # Save last model
        torch.save({
            'epoch': args.epochs,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'accuracy': current_acc,
        }, str(last_model_path))
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