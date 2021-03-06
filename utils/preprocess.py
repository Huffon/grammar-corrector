"""Data pre-processing function"""
import os
import re
import random
import logging
# import multiprocessing

import spacy

from tqdm import tqdm
from glob import glob
from typing import List

random.seed(42)
# NLP = spacy.load("en_core_web_sm")
# MANAGER = multiprocessing.Manager()
# WIKI_SRCS, WIKI_TGTS = MANAGER.list(), MANAGER.list()


# Ref: https://github.com/kakaobrain/helo_word/blob/master/gec/m2.py
def get_all_coder_ids(edits):
    coder_ids = set()
    for edit in edits:
        edit = edit.split("|||")
        coder_id = int(edit[-1])
        coder_ids.add(coder_id)
    coder_ids = sorted(list(coder_ids))
    return coder_ids


# Ref: https://github.com/kakaobrain/helo_word/blob/master/gec/m2.py
def m2_to_parallel(m2_files, ori, cor, drop_unchanged_samples, all):
    ori_fout = None
    if ori is not None:
        ori_fout = open(ori, 'w')
    cor_fout = open(cor, 'w')

    # Do not apply edits with these error types
    skip = {"noop", "UNK", "Um"}
    for m2_file in m2_files:
        entries = open(m2_file).read().strip().split("\n\n")
        for entry in entries:
            lines = entry.split("\n")
            ori_sent = lines[0][2:]  # Ignore "S "
            cor_tokens = lines[0].split()[1:]  # Ignore "S "
            edits = lines[1:]
            offset = 0

            coders = get_all_coder_ids(edits) if all == True else [0]
            for coder in coders:
                for edit in edits:
                    edit = edit.split("|||")
                    if edit[1] in skip: continue  # Ignore certain edits
                    coder_id = int(edit[-1])
                    if coder_id != coder: continue  # Ignore other coders
                    span = edit[0].split()[1:]  # Ignore "A "
                    start = int(span[0])
                    end = int(span[1])
                    cor = edit[2].split()
                    cor_tokens[start + offset:end + offset] = cor
                    offset = offset - (end - start) + len(cor)

                cor_sent = " ".join(cor_tokens)
                if drop_unchanged_samples and ori_sent == cor_sent:
                    continue

                if ori is not None:
                    ori_fout.write(ori_sent + "\n")
                cor_fout.write(cor_sent + "\n")


def preprocess_jfleg():
    """Preprocess JFLEG dataset
    """
    prefix = "jfleg"
    datasets = ["dev/dev", "test/test"]

    src, tgt = list(), list()

    with open("jfleg.src", "w") as src_out, open("jfleg.tgt", "w") as tgt_out:
        for dataset in datasets:
            f_src = open(os.path.join(prefix, f"{dataset}.src"), "r", encoding="utf-8")
            src_lines = f_src.readlines()

            refs = ["ref0", "ref1", "ref2", "ref3"]

            for ref in refs:
                f_tgt = open(os.path.join(prefix, f"{dataset}.{ref}"), "r", encoding="utf-8")
                tgt_lines = f_tgt.readlines()

                assert len(src_lines) == len(tgt_lines), "Source and Target should be parallel"

                for src, tgt in zip(src_lines, tgt_lines):
                    src_out.write(src)
                    tgt_out.write(tgt)


# def worker(srcs: List[str], tgts: List[str]):
#     """Multi-processing worker for WikEd preprocessing
#     """
#     for src, tgt in tqdm(zip(srcs, tgts), total=len(srcs)):
#         doc = NLP(src)
#         pos = [tok.pos_ for tok in doc]

#         if ("VERB" not in pos) or (len(pos) < 3):
#             continue

#         WIKI_SRCS.append(src)
#         WIKI_TGTS.append(tgt)


# def preprocess_wikiedit():
#     """Preprocess WikiEdit dataset
#     """
#     prefix = "wiked.tok."

#     with open(f"{prefix}err", "r") as f_src, open(f"{prefix}cor", "r") as f_tgt:
#         srcs = f_src.readlines()
#         tgts = f_tgt.readlines()

#         slices = len(srcs) // 8
#         process1 = multiprocessing.Process(target=worker, args=[srcs[:slices], tgts[:slices]])
#         process2 = multiprocessing.Process(target=worker, args=[srcs[slices:slices*2], tgts[slices:slices*2]])
#         process3 = multiprocessing.Process(target=worker, args=[srcs[slices*2:slices*3], tgts[slices*2:slices*3]])
#         process4 = multiprocessing.Process(target=worker, args=[srcs[slices*3:slices*4], tgts[slices*3:slices*4]])
        
#         process1.start()
#         process2.start()
#         process3.start()
#         process4.start()

#         process1.join()
#         process2.join()
#         process3.join()
#         process4.join()

#         pair = list(zip(WIKI_SRCS, WIKI_TGTS))
#         random.shuffle(pair)
#         src_lines, tgt_lines = zip(*pair)

#         split_ratio = int(len(src_lines) * 0.7)

#         train_src = src_lines[:split_ratio]
#         train_tgt = tgt_lines[:split_ratio]

#         valid_src = src_lines[split_ratio:]
#         valid_tgt = tgt_lines[split_ratio:]

#         with open("wiki_train.src", "w") as train_src_out, open("wiki_train.tgt", "w") as train_tgt_out:
#             for src, tgt in zip(train_src, train_tgt):
#                 train_src_out.write(src)
#                 train_tgt_out.write(tgt)

#         with open("wiki_valid.src", "w") as valid_src_out, open("wiki_valid.tgt", "w") as valid_tgt_out:
#             for src, tgt in zip(valid_src, valid_tgt):
#                 valid_src_out.write(src)
#                 valid_tgt_out.write(tgt)


def normalize(sentence):
    """Normalize sentence in LANG8 corpora
    """
    sentence = sentence.replace("`", "")
    sentence = sentence.replace("& nbsp ;", "")
    sentence = re.sub(" +", " ", sentence)
    return sentence


def preprocess_lang8():
    """Preprocess LANG8 dataset
    """
    nlp = spacy.load("en_core_web_sm")
    weirdos = ["(", ")", "{", "}", "[", "]", "<", ">", ":", "/"]
    
    with open("lang.src", "r") as f_src, open("lang.tgt", "r") as f_tgt:
        src_lines = f_src.readlines()
        tgt_lines = f_tgt.readlines()
        
        src_seen, tgt_seen = list(), list()
        new_src, new_tgt = list(), list()

        for src, tgt in tqdm(zip(src_lines, tgt_lines), total=len(src_lines)):
            if (src in src_seen) or tgt in tgt_seen:
                continue

            tags = [token.pos_ for token in nlp(src)]
            
            if "VERB" not in tags:
                continue

            first_src = src.split(" ")[0].lower()
            first_tgt = tgt.split(" ")[0].lower()

            if first_src != first_tgt:
                continue

            have_weirdo = False
            for weirdo in weirdos:
                if (weirdo in src) or (weirdo in tgt):
                    have_weirdo = True

            if have_weirdo:
                continue

            new_src.append(normalize(src))
            new_tgt.append(normalize(tgt))
            src_seen.append(src)
            tgt_seen.append(tgt)

    assert len(new_src) == len(new_tgt), "Source and Target should be parallel"
    
    pair = list(zip(new_src, new_tgt))
    random.shuffle(pair)
    src_lines, tgt_lines = zip(*pair)

    split_ratio = int(len(src_lines) * 0.85)

    train_src = src_lines[:split_ratio]
    train_tgt = tgt_lines[:split_ratio]

    valid_src = src_lines[split_ratio:]
    valid_tgt = tgt_lines[split_ratio:]

    with open("lang_train.src", "w") as train_src_out, open("lang_train.tgt", "w") as train_tgt_out:
        for src, tgt in zip(train_src, train_tgt):
            train_src_out.write(src)
            train_tgt_out.write(tgt)

    with open("lang_valid.src", "w") as valid_src_out, open("lang_valid.tgt", "w") as valid_tgt_out:
        for src, tgt in zip(valid_src, valid_tgt):
            valid_src_out.write(src)
            valid_tgt_out.write(tgt)


def preprocess_conll():
    """Preprocess CONLL 2014 dataset
    """
    nlp = spacy.load("en_core_web_sm")
    weirdos = ["(", ")", "{", "}", "[", "]", "<", ">", ":", "/", "http"]
    
    with open("conll2014.src", "r") as f_src, open("conll2014.tgt", "r") as f_tgt:
        src_lines = f_src.readlines()
        tgt_lines = f_tgt.readlines()
        
        new_src, new_tgt = list(), list()
        for src, tgt in tqdm(zip(src_lines, tgt_lines), total=len(src_lines)):
            tags = [token.pos_ for token in nlp(src)]
            
            if "VERB" not in tags:
                continue

            have_weirdo = False
            for weirdo in weirdos:
                if (weirdo in src) or (weirdo in tgt):
                    have_weirdo = True

            if have_weirdo:
                continue

            new_src.append(normalize(src))
            new_tgt.append(normalize(tgt))

    assert len(new_src) == len(new_tgt), "Source and Target should be parallel"
    
    pair = list(zip(new_src, new_tgt))
    random.shuffle(pair)
    src_lines, tgt_lines = zip(*pair)

    with open("conll_train.src", "w") as train_src_out, open("conll_train.tgt", "w") as train_tgt_out:
        for src, tgt in zip(src_lines, tgt_lines):
            train_src_out.write(src)
            train_tgt_out.write(tgt)


def create_pair(mode: str, datasets: List[str]):
    """Create source and target file using pre-generated list
    """
    src, tgt = list(), list()

    for dataset in datasets:
        src += open(f"{dataset}.src", "r").readlines()
        tgt += open(f"{dataset}.tgt", "r").readlines()

    pair = list(zip(src, tgt))
    random.shuffle(pair)
    src, tgt = zip(*pair)

    with open(f"data/{mode}.source", "w", encoding="utf-8") as f_src:
        for source in list(src):
            f_src.write(source)

    with open(f"data/{mode}.target", "w", encoding="utf-8") as f_tgt:
        for target in list(tgt):
            f_tgt.write(target)


def main():
    """Main function"""
    logging.info("[TRAIN] WI+Locness")
    m2_to_parallel(sorted(glob("wi+locness/m2/*.train.*m2")), "wi.src", "wi.tgt", False, True)

    logging.info("[TRAIN] FCE")
    m2_to_parallel(sorted(glob("fce/m2/*m2")), "fce.src", "fce.tgt", False, True)

    logging.info("[TRAIN] JFLEG")
    preprocess_jfleg()

    logging.info("[TRAIN] CoNLL 2013")
    m2_to_parallel(sorted(glob("release2.3.1/original/data/official-preprocessed.m2")), "2013.src", "2013.tgt", False, False)

    # logging.info("[TRAIN] WikEd")
    # preprocess_wikiedit()

    # LANG8 is not publicly released!
    # logging.info("[TRAIN] LANG8")
    # m2_to_parallel(sorted(glob("lang8*.m2")), "lang.src", "lang.tgt", True, True)
    # preprocess_lang8()

    # NUCLE is not publicly released!
    # logging.info("[TRAIN] NUCLE and CONLL 2014")
    # m2_to_parallel(sorted(glob("nucle*.m2")), "nucle.src", "nucle.tgt", True, True)
    # m2_to_parallel(sorted(glob("conll14st-preprocessed.m2")), "conll2014.src", "conll2014.tgt", False, False)
    # preprocess_conll()
    create_pair("train", ["wi", "fce", "jfleg", "2013", "lang_train", "nucle", "conll_train"])

    logging.info("[VAL] WI+Locness")
    m2_to_parallel(sorted(glob("wi+locness/m2/*.dev.*m2")), "wi_test.src", "wi_test.tgt", False, True)

    logging.info("[VAL] CoNLL 2014")
    m2_to_parallel(sorted(glob("conll14st-test-data/noalt/official-2014.combined.m2")), "2014.src", "2014.tgt", False, False)
    create_pair("val", ["wi_test", "2014", "lang_valid"])

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
