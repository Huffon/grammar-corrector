"""Generation using BART"""
import torch
from fairseq.models.bart import BARTModel


def main():
    bart = BARTModel.from_pretrained("ckpt", checkpoint_file="checkpoint_best.pt")
    bart.cuda()
    bart.half()
    bart.eval()

    with open("output/input.txt") as source:
        lines = source.readlines()
        lines = [line.replace("\n", "") for line in lines]

        with torch.no_grad():
            preds = bart.sample(lines)

            for i, (line, pred) in enumerate(zip(lines, preds)):
                pred = pred.replace("&apos;", "'")
                print(f"[ori] ({i+1}): {line}")
                print(f"[cor] ({i+1}): {pred}")
                print()


if __name__ == "__main__":
    main()
