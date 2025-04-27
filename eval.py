#! /usr/bin/env python
#! coding:utf-8
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import torch.nn as nn
import argparse
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import logging
import sys
import time

from dataloader.jhmdb_loader import load_jhmdb_data, Jdata_generator, JConfig
from dataloader.shrec_loader import load_shrec_data, Sdata_generator, SConfig
# from models.DDNet_Original import DDNet_Original as DDNet

from models.DDNet_Original_ODE import DDNet_Original as DDNet

from utils import makedir

def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    np.random.seed(seed)

def count_parameters(model):
    """Count the number of trainable parameters in the model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def print_model_summary(model, input_shape1, input_shape2):
    """Print model summary including parameter count and layer information"""
    total_params = count_parameters(model)
    print("\nModel Summary:")
    print("=" * 50)
    print(f"Total trainable parameters: {total_params:,}")
    print("\nLayer Information:")
    print("-" * 50)
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv1d, nn.Linear, nn.BatchNorm1d)):
            params = sum(p.numel() for p in module.parameters() if p.requires_grad)
            print(f"{name}: {module.__class__.__name__} - Parameters: {params:,}")
    print("=" * 50)
    return total_params


def evaluate(model, device, test_loader, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for _, (data1, data2, target) in enumerate(tqdm(test_loader)):
            M, P, target = data1.to(device), data2.to(device), target.to(device)
            output = model(M, P)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            
            all_predictions.extend(pred.cpu().numpy().flatten())
            all_targets.extend(target.cpu().numpy())

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    
    return test_loss, accuracy, all_predictions, all_targets

def main():
    parser = argparse.ArgumentParser(description='Evaluate DDNet model')
    parser.add_argument('--model-path', type=str, required=True,
                      help='path to the trained model checkpoint')
    parser.add_argument('--batch-size', type=int, default=64,
                      help='input batch size for evaluation (default: 64)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                      help='disables CUDA evaluation')
    parser.add_argument('--seed', type=int, default=42,
                      help='random seed (default: 42)')
    parser.add_argument('--dataset', type=int, required=True,
                      help='0 for JHMDB, 1 for SHREC coarse, 2 for SHREC fine')
    parser.add_argument('--output-dir', type=str, default='evaluation_results',
                      help='directory to save evaluation results')
    args = parser.parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Setup device
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    makedir(output_dir)
    logging.basicConfig(filename=output_dir/'eval.log', level=logging.INFO)
    
    # Load dataset
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

    # Load test data
    _, Test, le = load_data()
    X_0_t, X_1_t, Y_t = data_generator(Test, Config, le)
    X_0_t = torch.from_numpy(X_0_t).to(torch.float32)
    X_1_t = torch.from_numpy(X_1_t).to(torch.float32)
    Y_t = torch.from_numpy(Y_t).to(torch.long)

    testset = torch.utils.data.TensorDataset(X_0_t, X_1_t, Y_t)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size)

    # Initialize model
    model = DDNet(Config.frame_l, Config.joint_n, Config.joint_d,
                 Config.feat_d, Config.filters, clc_num)
    
    # Load model checkpoint
    checkpoint = torch.load(args.model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    # Print model summary
    input_shape1 = (1, Config.frame_l, Config.feat_d)  # Shape for M input
    input_shape2 = (1, Config.frame_l, Config.joint_n, Config.joint_d)  # Shape for P input
    total_params = print_model_summary(model, input_shape1, input_shape2)
    
    criterion = nn.CrossEntropyLoss(reduction='sum')
    
    # Evaluate model
    test_loss, accuracy, predictions, targets = evaluate(model, device, test_loader, criterion)
    
    # Print and log results
    results = f"""
    Evaluation Results:
    -------------------
    Test Loss: {test_loss:.4f}
    Accuracy: {accuracy:.2f}%
    Total Parameters: {total_params:,}
    """
    print(results)
    logging.info(results)
    
    # Generate and save confusion matrix
    cm = confusion_matrix(targets, predictions)
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.savefig(str(output_dir / 'confusion_matrix.png'))
    
    # Save classification report
    report = classification_report(targets, predictions)
    with open(output_dir / 'classification_report.txt', 'w') as f:
        f.write(report)
    
    # Calculate and save per-class accuracy
    class_accuracy = cm.diagonal() / cm.sum(axis=1)
    np.save(output_dir / 'class_accuracy.npy', class_accuracy)
    
    # Save detailed results
    results_dict = {
        'test_loss': test_loss,
        'accuracy': accuracy,
        'predictions': predictions,
        'targets': targets,
        'confusion_matrix': cm,
        'class_accuracy': class_accuracy,
        'total_parameters': total_params
    }
    np.save(output_dir / 'evaluation_results.npy', results_dict)

if __name__ == '__main__':
    main() 