UnsupNTS
==============

This is an implementation of the Unsupervised Neural Text Simplification system and their semi-supervised variants mentioned in the paper:

Sai Surya, Abhijit Mishra, Anirban Laha, Parag Jain, and Karthik Sankaranarayanan. **[Unsupervised Neural Text Simplification](https://arxiv.org/pdf/1810.07931.pdf)** arXiv preprint arXiv:1810.07931 (2018).


Requirements
--------
- Python 3.6
- PyTorch 1.0.1
- NLTK 
- textstat 
- fuzzywuzzy


To Reproduce Results
--------
Get the data needed from (link) and extract the zip file
```
unzip tsdata.zip
```
`tsdata.zip` has
- `fkdifficpart-2m-1.lower` and `fkeasypart-2m-1.lower`: Unsupervised sets of easy and difficult set of sentences judged on readability ease scores.
- `fk.lower.vec`: Dict2vec embeddings trained on the above unsupervised sets. 
- `wiki-split.en.lower` and `wiki-split.sen.lower`: 10k parallel pairs of difficult and its simplified sentence.

Train the models using
```
bash train.sh
```
`train.sh` has 
- UNTS system from unsupervised simplification data using the exact same settings described in the paper.
- UNTS-10k system, using additional 10k supervised pairs of mixture of split-rephrase and simplification parallel pairs. 
- UNMT system on the unsupervised simplification data.
- ablations on adversarial and separation/classifier losses

For the above systems, training takes about 6 hrs in Nvidia T4 GPU. For more details and additional options, run the above scripts with the `--help` flag.
Alternatively, visit the ipynb in google colaboratory to reproduce the results.

Generation and Evaluation of Simplifications 
--------
```
bash translate.sh
```
The above command does
- For generating simplifications of `INPUT.TXT`
- For evaluation of `OUTPUT.TXT`, filter the redundancies
- Compute stand alone metrics such as fk score difference, tree similarity and document similarity metrics
- Compute  SARI, BLEU and word-diff

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
