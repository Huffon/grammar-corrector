# Optional
cp -r nucle/* /workspace/grammar-corrector/
cp -r lang8/* /workspace/grammar-corrector/

# WI+Locness
wget https://www.cl.cam.ac.uk/research/nl/bea2019st/data/wi+locness_v2.1.bea19.tar.gz
tar xvzf wi+locness_v2.1.bea19.tar.gz
rm wi+locness_v2.1.bea19.tar.gz

# FCE
wget https://www.cl.cam.ac.uk/research/nl/bea2019st/data/fce_v2.1.bea19.tar.gz
tar xvzf fce_v2.1.bea19.tar.gz
rm fce_v2.1.bea19.tar.gz

# JFLEG
git clone https://github.com/keisks/jfleg.git

# CoNLL 2013
wget https://www.comp.nus.edu.sg/~nlp/conll13st/release2.3.1.tar.gz
tar xvzf release2.3.1.tar.gz
rm release2.3.1.tar.gz

# CoNLL 2014
wget https://www.comp.nus.edu.sg/~nlp/conll14st/conll14st-test-data.tar.gz
tar xvzf conll14st-test-data.tar.gz
rm conll14st-test-data.tar.gz

# WikiEdit
# wget http://data.statmt.org/romang/wiked/wiked-v1.0.en.prepro.tgz
# tar xvzf wiked-v1.0.en.prepro.tgz
# rm wiked-v1.0.en.prepro.tgz

mkdir data
python utils/preprocess.py

rm -rf wi+locness fce release2.3.1 conll14st-test-data jfleg
rm *.src *.tgt

wget -N 'https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/encoder.json'
wget -N 'https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/vocab.bpe'
wget -N 'https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/dict.txt'

for SPLIT in train val
do
  for LANG in source target
  do
    python utils/multiprocessing_bpe_encoder.py \
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
