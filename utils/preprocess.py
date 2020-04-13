"""Data pre-processing function"""
import os
import random
import logging
from glob import glob
from typing import List

random.seed(42)


# src: https://github.com/kakaobrain/helo_word/blob/master/gec/m2.py
def get_all_coder_ids(edits):
    coder_ids = set()
    for edit in edits:
        edit = edit.split("|||")
        coder_id = int(edit[-1])
        coder_ids.add(coder_id)
    coder_ids = sorted(list(coder_ids))
    return coder_ids


# src: https://github.com/kakaobrain/helo_word/blob/master/gec/m2.py
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
    m2_to_parallel(sorted(glob("wi+locness/m2/*m2")), "wi.src", "wi.tgt", False, True)

    logging.info("[TRAIN] FCE")
    m2_to_parallel(sorted(glob("fce/m2/*m2")), "fce.src", "fce.tgt", False, True)

    logging.info("[TRAIN] JFLEG")
    preprocess_jfleg()
    create_pair("train", ["wi", "fce", "jfleg"])

    logging.info("[VAL] CoNLL 2013")
    m2_to_parallel(sorted(glob("release2.3.1/original/data/official-preprocessed.m2")), "2013.src", "2013.tgt", False, False)

    logging.info("[VAL] CoNLL 2014")
    m2_to_parallel(sorted(glob("conll14st-test-data/noalt/official-2014.combined.m2")), "2014.src", "2014.tgt", False, False)
    create_pair("val", ["2013", "2014"])

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
