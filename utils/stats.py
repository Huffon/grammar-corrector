"Data statistics"
def main():
    with open("data/train.source", "r") as f_src, open("data/train.target", "r") as f_tgt:
        src_lines = f_src.readlines()
        tgt_lines = f_tgt.readlines()

        src_max = len(max(src_lines, key=len))
        tgt_max = len(max(tgt_lines, key=len))

        src_min = len(min(src_lines, key=len))
        tgt_min = len(min(tgt_lines, key=len))

        src_tmp = [len(line) for line in src_lines] 
        src_avg = float(sum(src_tmp)) / len(src_tmp)

        tgt_tmp = [len(line) for line in tgt_lines] 
        tgt_avg = float(sum(tgt_tmp)) / len(tgt_tmp)

        print(f"[SRC] Max: {src_max}\tMin: {src_min}\tAvg: {int(src_avg)}")
        print(f"[TGT] Max: {tgt_max}\tMin: {tgt_min}\tAvg: {int(tgt_avg)}")


if __name__ == "__main__":
    main()