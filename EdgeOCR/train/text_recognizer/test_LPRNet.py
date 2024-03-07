import argparse
import os
import time
import torch
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from strsimpy.normalized_levenshtein import NormalizedLevenshtein

# Assume these imports are from custom modules
from data.load_data import CHARS, DataLoader
from model.OCRNet import build_OCRNet

def get_parser():
    parser = argparse.ArgumentParser(description='parameters to test model')
    parser.add_argument('--img_size', default=[240, 75], help='the image size')
    parser.add_argument('--test_img_dirs', default="/dataset", help='the test images path')
    parser.add_argument('--lpr_max_len', default=30, type=int, help='license plate number max length')
    parser.add_argument('--test_batch_size', type=int, default=1, help='testing batch size')
    parser.add_argument('--pretrained_model', default='./weights/Final_OCRNet_model.pth', help='pretrained model path')
    parser.add_argument('--cpu', action='store_true', help='Use CPU to test model')
    return parser.parse_args()

def collate_fn(batch):
    imgs = [torch.from_numpy(sample[0]) for sample in batch]
    labels = [label for sample in batch for label in sample[1]]
    lengths = [sample[2] for sample in batch]
    return torch.stack(imgs, 0), torch.tensor(labels), lengths

def test():
    args = get_parser()
    device = torch.device("cpu" if args.cpu else "cuda:0")

    OCRNet = build_OCRNet(lpr_max_len=args.lpr_max_len, phase_train=False, class_num=len(CHARS))
    OCRNet.to(device)
    print("Successfully built network!")

    if args.pretrained_model:
        OCRNet.load_state_dict(torch.load(args.pretrained_model, map_location=device))
        print("Loaded pretrained model successfully!")
    else:
        print("[Error] Can't find pretrained model, please check!")
        return False

    test_dataset = DataLoader(args.test_img_dirs.split(','), args.img_size, args.lpr_max_len, train=False)
    evaluate(OCRNet, test_dataset, args)

def evaluate(net, dataset, args):
    net.eval()
    loader = DataLoader(dataset, args.test_batch_size, shuffle=False, num_workers=2, collate_fn=collate_fn)
    
    correct = 0
    total = 0
    norm_lev = NormalizedLevenshtein()
    total_similarity = 0
    
    t1 = time.time()
    with torch.no_grad():
        for images, labels, lengths in tqdm(loader):
            images = images.to(next(net.parameters()).device)
            logits = net(images)
            preds = logits.argmax(1).permute(1, 0).cpu().numpy()
            
            for pred, label, length in zip(preds, labels, lengths):
                pred_label = ''.join([CHARS[c] for c in pred[:length] if c != len(CHARS) - 1])
                true_label = ''.join([CHARS[c] for c in label[:length]])
                
                similarity = norm_lev.similarity(pred_label, true_label)
                total_similarity += similarity
                
                if pred_label == true_label:
                    correct += 1
                total += 1
                
                print(f"Predicted: {pred_label}, True: {true_label}, Similarity: {similarity:.4f}")

    accuracy = correct / total
    avg_similarity = total_similarity / total
    t2 = time.time()
    
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Average Levenshtein Similarity: {avg_similarity:.4f}")
    print(f"Test Speed: {(t2 - t1) / len(dataset):.4f}s per image")

if __name__ == "__main__":
    test()