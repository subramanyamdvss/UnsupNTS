UnsupNTS
==============

This is an implementation of the Unsupervised Neural Text Simplification system and their semi-supervised variants mentioned in the paper:

Sai Surya, Abhijit Mishra, Anirban Laha, Parag Jain, and Karthik Sankaranarayanan. **[Unsupervised Neural Text Simplification](https://arxiv.org/pdf/1810.07931.pdf)** arXiv preprint arXiv:1810.07931 (2018).

![](diagram-1.png | width=48)

Requirements
--------
- Python 3.6
- PyTorch 1.0.1
- NLTK 
- textstat 
- fuzzywuzzy


Training
--------
Download `tsdata.zip` from **[link](https://drive.google.com/open?id=1oHDTOX5u4JS8RvnvlogeQaGPvarjKRk-)** and extract
```
unzip tsdata.zip
```
`tsdata.zip` has
- `fkdifficpart-2m-1.lower` and `fkeasypart-2m-1.lower`: Unsupervised sets of easy and difficult set of sentences judged on readability ease scores.
- `fk.lower.vec`: Dict2vec embeddings trained on the above unsupervised sets. 
- `wiki-split.en.lower` and `wiki-split.sen.lower`: 10k parallel pairs of difficult and simplified variants.
- `test.en` and `references.tsv`: Test set and references (eight tab seperated references per each sentence in `test.en`).

Train the models using
```
bash train.sh
```
`train.sh` has 
- UNTS system from unsupervised simplification data using the exact same settings described in the paper.
- UNTS-10k system, using additional 10k supervised pairs of mixture of split-rephrase and simplification parallel pairs. 
- UNMT system on the unsupervised simplification data.
- ablations on adversarial and separation/classifier losses.

For the above systems, training takes about 6 hrs each in Nvidia T4 GPU. For more details and additional options, run the above scripts with the `--help` flag.
Alternatively, visit the **[ipynb](https://drive.google.com/file/d/1cVuzsU389WC9-1NliaP6mpBU77ZkgW6v/view?usp=sharing)** in google colaboratory to reproduce the results. To access pretrained models visit **[link](https://drive.google.com/file/d/11U-MnbjkLQXK_z5R6RPsfSZWwmSPoj34/view?usp=sharing)**. The folder `predictions` has the generations from the pretrained models. 

**Note**: Pretrained models were trained with pytorch 0.3.1 and may not exactly reproduce the result in pytorch 1.0.1 .

Generation and Evaluation of Simplifications 
--------
```
bash translate.sh
```
`translate.sh` is used for
- Generating simplifications of `test.en`.
- Computing stand alone metrics such as Flesch readability ease score difference, Tree similarity and Document similarity metrics.
- Computing  SARI, BLEU and Word-diff metrics.

Acknowledgements
--------
A large portion of this repo is borrowed from the following repos: https://github.com/artetxem/undreamt and https://github.com/senisioi/NeuralTextSimplification.

If you use this code for academic research, please cite the paper in question:
```
@article{surya2018unsupervised,
  title={Unsupervised Neural Text Simplification},
  author={Surya, Sai and Mishra, Abhijit and Laha, Anirban and Jain, Parag and Sankaranarayanan, Karthik},
  journal={arXiv preprint arXiv:1810.07931},
  year={2018}
}
```
