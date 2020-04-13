# WI+Locness
wget https://www.cl.cam.ac.uk/research/nl/bea2019st/data/wi+locness_v2.1.bea19.tar.gz
tar xvzf wi+locness_v2.1.bea19.tar.gz
rm wi+locness_v2.1.bea19.tar.gz

# FCE
wget https://www.cl.cam.ac.uk/research/nl/bea2019st/data/fce_v2.1.bea19.tar.gz
tar xvzf fce_v2.1.bea19.tar.gz
rm fce_v2.1.bea19.tar.gz

rm -rf wi+locness fce

python utils/preprocess.py

wget -N 'https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/encoder.json'
wget -N 'https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/vocab.bpe'
wget -N 'https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/dict.txt'

for SPLIT in train val
do
  for LANG in source target
  do
    python -m examples.roberta.multiprocessing_bpe_encoder \
    --encoder-json encoder.json \
    --vocab-bpe vocab.bpe \
    --inputs "data/$SPLIT.$LANG" \
    --outputs "data/$SPLIT.bpe.$LANG" \
    --workers 60 \
    --keep-empty;
  done
done

fairseq-preprocess \
    --source-lang source \
    --target-lang target \
    --trainpref data/train.bpe \
    --validpref data/val.bpe \
    --destdir data-bin \
    --workers 60 \
    --srcdict dict.txt \
    --tgtdict dict.txt