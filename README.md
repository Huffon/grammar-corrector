# Grammar Error Corrector

This repository contains **Grammar Error Corrector** API trained using **BART** architecture

Lots of code are borrowed from [fairseq](https://github.com/pytorch/fairseq) library

<br/>

## Requirements

- **Python** version >= 3.7
- [PyTorch](https://pytorch.org/get-started/locally/) version >= 1.4.0
- [fairseq](https://github.com/pytorch/fairseq) >= 0.9.0

```
conda create -n corrector python=3.7
conda activate corrector
conda install pytorch cudatoolkit=10.1 -c pytorch
pip install fairseq requests tqdm
```

<br/>

## Usage

- To **download** and **preprocess** data, run following command:

```
bash preprocess.sh
```

### BART

- To **download** and **fine-tune** pre-trained **BART**, run following command:

```
wget https://dl.fbaipublicfiles.com/fairseq/models/bart.large.tar.gz
tar xvzf bart.large.tar.gz
bash train.sh
```

- To **generate** example sentence using fine-tuned **BART**, run following command:

```
vi output/input.txt
python generate.py
```


<br/>

## Example

- To **test** your own sentences, fill [**input.txt**](output/input.txt) with your sentences

```
[ori] (1): I went for school.
[cor] (1): Then I went to school.

[ori] (2): He don't speak English.
[cor] (2): He doesn't speak English.

[ori] (3): I doens't have money.
[cor] (3): I don't have money.

[ori] (4): I want going for Guam.
[cor] (4): I want to go to Guam.

[ori] (5): I like here.
[cor] (5): I like it here.

[ori] (6): He will happy.
[cor] (6): He will be happy.

[ori] (7): I is happy.
[cor] (7): I am happy.

...

[ori] (15): I want to school.
[cor] (15): I want to go to school.

[ori] (16): I no speak English.
[cor] (16): No, I don't speak English.

[ori] (17): I have two foots.
[cor] (17): I have two feet.

[ori] (18): Despite it was noisy and crowded, we had a good time.
[cor] (18): Despite it being noisy and crowded, we had a good time.
```

<br/>

## Data statistics (w. LANG8 & NUCLE, w/o. WikEd)

```
Source length
    Max: 1095
    Min: 2
    Avg: 80

Target length
    Max: 1153
    Min: 1
    Avg: 81
```

<br/>

## References
- [BEA 2019](https://convention2.allacademic.com/one/bea/bea19/)
- [CoNLL 2013](https://www.comp.nus.edu.sg/~nlp/conll13st.html)
- [CoNLL 2014](https://www.comp.nus.edu.sg/~nlp/conll14st.html)
- [JFLEG](https://github.com/keisks/jfleg)
- [fairseq](https://github.com/pytorch/fairseq)
- [Kakao Brains' **helo_word**](https://github.com/kakaobrain/helo_word)
