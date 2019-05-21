UnsupNTS
==============

This is an implementation of the Unsupervised Neural Text Simplification system and their semi-supervised variants mentioned in the paper:

Sai Surya, Abhijit Mishra, Anirban Laha, Parag Jain, and Karthik Sankaranarayanan. **[Unsupervised Neural Text Simplification](https://arxiv.org/pdf/1810.07931.pdf)** arXiv preprint arXiv:1810.07931 (2018).


Requirements
--------
- Python 3.6
- PyTorch (tested with v1.0.1) 
- NLTK
- textstat
- fuzzywuzzy

Alternatively in google colab:
```
!pip install fuzzywuzzy
import nltk
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')
!pip2 install textstat 
!pip install textstat
```

Train the models (to reproduce our results)
--------

The following command trains a UNTS system from unsupervised simplification data using the exact same settings described in the paper.

```
python3 undreamt/train.py --src_embeddings EMBEDD.VEC --trg_embeddings EMBEDD.VEC  --save MODEL_PREFIX \
 --cuda --disable_backtranslation --unsup --enable_mgan --add_control --easyprefix EASY.TXT \
  --difficprefix DIFFIC.TXT 
```
The following command trains a UNTS-10k system, using additional 10k supervised pairs of mixture of split-rephrase and simplification parallel pairs. 
```
python3 undreamt/train.py --src_embeddings EMBEDD.VEC --trg_embeddings EMBEDD.VEC  --save MODEL_PREFIX \
--cuda --disable_backtranslation --src2trg DIFFIC_PARALLEL.TXT EASY_PARALLEL.TXT --trg2src EASY_PARALLEL.TXT DIFFIC_PARALLEL.TXT \
--enable_mgan --add_control --easyprefix EASY.TXT --difficprefix DIFFIC.TXT 
```
The following command trains a UNMT system on the unsupervised simplification data.
```
python3 undreamt/train.py --src DIFFIC.TXT --trg EASY.TXT --src_embeddings EMBEDD.VEC --trg_embeddings EMBEDD.VEC  --save MODEL_PREFIX --batch 32 --cuda --unsup 
```
For the above systems, training takes about 6 hrs in Nvidia T4 GPU. For more details and additional options, run the above scripts with the `--help` flag.

Data
--------
- `EASY.TXT` and `DIFFIC.TXT`: unsupervised sets of easy and difficult set of sentences judged on readability ease scores.
- `EMBEDD.VEC`: dict2vec embeddings trained on the above unsupervised sets. 
- `EASY_PARALLEL.TXT` and `DIFFIC_PARALLEL.TXT`: 10k parallel pairs of difficult and its simplified sentence.

Generation and Evaluation of Simplifications 
--------
For generating simplifications of `INPUT.TXT`
```
python3 undreamt/translate.py MODEL_PREFIX --input INPUT.TXT --output OUTPUT.TXT  --noise 0.0
```
For evaluation of `OUTPUT.TXT`, filter the redundancies
```
python3  gendirprod/noredund.py < OUTPUT.TXT > OUTPUT.TXT
```
Compute stand alone metrics such as fk score difference, tree similarity and document similarity metrics
```
python3 fk_ts_ds.py -i OUTPUT.TXT -src INPUT.TXT 
```
Compute  SARI, BLEU and word-diff, where `GENF` is a folder with `OUTPUT.TXT`
```
python2 evaluate.py INPUT.TXT REFERENCES.TSV GENF 
```
Acknowledgements
--------
A large portion of this repo is borrowed from the following repos: https://github.com/artetxem/undreamt and https://github.com/senisioi/NeuralTextSimplification 

If you use this code for academic research, please cite the paper in question:
```
@article{surya2018unsupervised,
  title={Unsupervised Neural Text Simplification},
  author={Surya, Sai and Mishra, Abhijit and Laha, Anirban and Jain, Parag and Sankaranarayanan, Karthik},
  journal={arXiv preprint arXiv:1810.07931},
  year={2018}
}
```
