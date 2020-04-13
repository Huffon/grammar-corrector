# Grammar Error Corrector

This repository contains Grammar Error Corrector API trained using **BART** architecture

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
pip install fairseq
```

<br/>

## Usage

- To **download** and **preprocess** data, run following command:

```
bash preprocess.sh
```

### BART

- To download and fine-tune pre-trained **BART**, run following command:

```
wget https://dl.fbaipublicfiles.com/fairseq/models/bart.large.tar.gz
tar xvzf bart.large.tar.gz
bash train.sh
```

- To **generate** example sentence using [fine-tunied BART](), run following command:

```
python generate.py
```


<br/>

## Example

- To test your own sentences, fill [**input.txt**](output/input.txt) with your sentences

```

```

<br/>

## References
- [BEA 2019](https://convention2.allacademic.com/one/bea/bea19/)
- [CoNLL 2014](https://www.comp.nus.edu.sg/~nlp/conll14st.html)
- [fairseq](https://github.com/pytorch/fairseq)