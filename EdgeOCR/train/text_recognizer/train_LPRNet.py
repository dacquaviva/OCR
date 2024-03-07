import argparse
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
import numpy as np
from tqdm import tqdm
from strsimpy.normalized_levenshtein import NormalizedLevenshtein

# Assume these imports are from custom modules
from data.load_data import CHARS, DataLoader
from model.OCRNet import build_OCRNet

def get_parser():
    parser = argparse.ArgumentParser(description='parameters to train net')
    parser.add_argument('--max_epoch', type=int, default=1000)
    parser.add_argument('--img_size', nargs=2, type=int, default=[240, 75])
    parser.add_argument('--train_img_dirs', default="/dataset")
    parser.add_argument('--test_img_dirs', default="/dataset")
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--lpr_max_len', default=30, type=int)
    parser.add_argument('--train_batch_size', type=int, default=4)
    parser.add_argument('--test_batch_size', type=int, default=4)
    parser.add_argument('--save_folder', default='./runs/')
    parser.add_argument('--cpu', action='store_true', help='Use CPU to train model')
    return parser.parse_args()

def sparse_tuple_for_ctc(T_length, lengths):
    return tuple([T_length] * len(lengths)), tuple(lengths)

def collate_fn(batch):
    imgs = [torch.from_numpy(sample[0]) for sample in batch]
    labels = [label for sample in batch for label in sample[1]]
    lengths = [sample[2] for sample in batch]
    return torch.stack(imgs, 0), torch.tensor(labels), lengths

def train():
    args = get_parser()
    device = torch.device("cpu" if args.cpu else "cuda:0")

    OCRNet = build_OCRNet(lpr_max_len=args.lpr_max_len, phase_train=True, class_num=len(CHARS), device=device)
    OCRNet.to(device)

    optimizer = optim.Adam(OCRNet.parameters(), lr=args.learning_rate)
    ctc_loss = nn.CTCLoss(blank=len(CHARS)-1, reduction='mean')

    train_dataset = DataLoader(args.train_img_dirs.split(','), args.img_size, args.lpr_max_len)
    test_dataset = DataLoader(args.test_img_dirs.split(','), args.img_size, args.lpr_max_len, train=False)

    best_acc = 0
    for epoch in range(args.max_epoch):
        OCRNet.train()
        epoch_loss = 0
        batch_iterator = iter(DataLoader(train_dataset, args.train_batch_size, shuffle=True, collate_fn=collate_fn))

        pbar = tqdm(range(len(train_dataset) // args.train_batch_size), desc=f'Epoch {epoch+1}/{args.max_epoch}')
        for _ in pbar:
            images, labels, lengths = next(batch_iterator)
            images = images.to(device)
            labels = labels.to(device)

            logits = OCRNet(images)
            log_probs = logits.permute(2, 0, 1).log_softmax(2)
            input_lengths, target_lengths = sparse_tuple_for_ctc(logits.size(1), lengths)
            loss = ctc_loss(log_probs, labels, input_lengths, target_lengths)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        print(f'Epoch [{epoch+1}/{args.max_epoch}], Loss: {epoch_loss/(len(train_dataset) // args.train_batch_size):.4f}')

        if (epoch + 1) % 10 == 0:
            acc = evaluate(OCRNet, test_dataset, args)
            if acc > best_acc:
                best_acc = acc
                torch.save(OCRNet.state_dict(), os.path.join(args.save_folder, 'best_OCRNet.pth'))

    print(f"Best test accuracy: {best_acc:.4f}")

def evaluate(net, dataset, args):
    net.eval()
    correct = 0
    total = 0
    NormLev = NormalizedLevenshtein()
    
    with torch.no_grad():
        for images, labels, lengths in DataLoader(dataset, args.test_batch_size, collate_fn=collate_fn):
            images = images.to(next(net.parameters()).device)
            logits = net(images)
            preds = logits.argmax(1).permute(1, 0).cpu().numpy()
            
            for pred, label, length in zip(preds, labels, lengths):
                pred_label = ''.join([CHARS[c] for c in pred[:length] if c != len(CHARS) - 1])
                true_label = ''.join([CHARS[c] for c in label[:length]])
                similarity = NormLev.similarity(pred_label, true_label)
                if similarity > 0.9:  # Adjust threshold as needed
                    correct += 1
                total += 1

    accuracy = correct / total
    print(f"Test Accuracy: {accuracy:.4f}")
    return accuracy

if __name__ == "__main__":
    train()