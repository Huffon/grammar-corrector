"""Generation using BART"""
import torch
from fairseq.models.bart import BARTModel


def reorder(sent: str):
    """Post-process automatically corrected sentence
    """
    specials = [".", ",", "?", "!", "n't", "'s", "'ve"]
    result = list()
    
    sent = sent.replace("&apos;", "'")
    tokens = sent.split(" ")

    if tokens[0] == "-":
        tokens.pop(0)

    for token in tokens:
        if token in specials:
            result[-1] = result[-1] + token
            continue

        if len(result) > 0 and (result[-1] in ["'", '"']):
            result[-1] = result[-1] + token
        result.append(token)

    return " ".join(result)


def main():
    bart = BARTModel.from_pretrained("ckpt", checkpoint_file="checkpoint8.pt")
    bart.cuda()
    bart.half()
    bart.eval()

    with open("output/input.txt") as source:
        lines = source.readlines()
        lines = [line.replace("\n", "") for line in lines]

        with torch.no_grad():
            preds = bart.sample(lines)

            for i, (line, pred) in enumerate(zip(lines, preds)):
                print(f"[ori] ({i+1}): {line}")
                print(f"[cor] ({i+1}): {reorder(pred)}")
                print()


if __name__ == "__main__":
    main()
