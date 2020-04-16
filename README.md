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

- Best loss: **2.75** (**1** epoch)
- To **download** and **fine-tune** pre-trained **BART**, run following command:

```
wget https://dl.fbaipublicfiles.com/fairseq/models/bart.large.tar.gz
tar xvzf bart.large.tar.gz
bash train.sh
```

- To **generate** example sentence using fine-tuned **BART**, run following command:

```
python generate.py
```


<br/>

## Example

- To **test** your own sentences, fill [**input.txt**](output/input.txt) with your sentences

```
[Before]
(1): He am happy.
(2): He will happy.
(3): I went for school.
(4): He don't speak English.
(5): I doens't have money.
(6): I want going for Guam.
(7): I like here.

[After]
(1): He is happy.
(2): He will be happy.
(3): Then I went to school.
(4): He doesn't speak English.
(5): I do n't have money.
(6): I want to go to Guam.
(7): I like it here.
```

<br/>

## Data statistics (w/o WikEd)

```
Source length
    Max: 1095
    Min: 2
    Avg: 86

Target length
    Max: 1153
    Min: 1
    Avg: 87
```

<br/>

## References
- [BEA 2019](https://convention2.allacademic.com/one/bea/bea19/)
- [CoNLL 2013](https://www.comp.nus.edu.sg/~nlp/conll13st.html)
- [CoNLL 2014](https://www.comp.nus.edu.sg/~nlp/conll14st.html)
- [JFLEG](https://github.com/keisks/jfleg)
- [fairseq](https://github.com/pytorch/fairseq)
- [Kakao Brains' **helo_word**](https://github.com/kakaobrain/helo_word)
