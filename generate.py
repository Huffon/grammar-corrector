"""Generation using BART"""
import torch
from fairseq.models.bart import BARTModel


def main():
    bart = BARTModel.from_pretrained('ckpt', checkpoint_file='checkpoint_best.pt')
    bart.cuda()
    bart.half()
    bart.eval()

    with open('output/input.txt') as source:
        lines = source.readlines()
        lines = [line.replace("\n", "") for line in lines]

        print("[Before]")
        for i, line in enumerate(lines):
            print(f"({i+1}): {line}")
        
        with torch.no_grad():
            preds = bart.sample(lines, temperature=0.8)
            print("\n[After]")
            for i, pred in enumerate(preds):
                pred = pred.replace("&apos;", "'")
                print(f"({i+1}): {pred}")


if __name__ == "__main__":
    main()
