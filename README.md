UnsupNTS: Unsupervised Neural Text Simplification
==============

This is the original implementation of the Unsupervised Neural Text Simplification system and their semi-supervised variants mentioned in the ACL 2019 long paper:

Sai Surya, Abhijit Mishra, Anirban Laha, Parag Jain, and Karthik Sankaranarayanan. **[Unsupervised Neural Text Simplification](https://arxiv.org/pdf/1810.07931.pdf)** arXiv preprint arXiv:1810.07931 (2018).

<img src="UnsupNTS.png" width="500">


Training
--------
Download `tsdata.zip` from **[link](https://drive.google.com/open?id=1oHDTOX5u4JS8RvnvlogeQaGPvarjKRk-)** and extract
```
unzip tsdata.zip
```
`tsdata.zip` has
- Unsupervised sets of easy and difficult set of sentences judged on readability ease scores.
- Dict2vec embeddings trained on the above unsupervised sets. 
- 10k parallel pairs of difficult and simplified variants.
- Test set and references - eight tab seperated references per each test sentence.

Train the models using
```
bash train.sh
```
`train.sh` has 
- UNTS system from unsupervised simplification data using the exact same settings described in the paper.
- UNTS-10k system, using additional 10k supervised pairs of mixture of split-rephrase and simplification parallel pairs. 
- UNMT system on the unsupervised simplification data.
- ablations on adversarial and separation/classifier losses.

For more details and additional options, run the above scripts with the `--help` flag.
Alternatively, visit the **[ipynb](https://drive.google.com/file/d/1cVuzsU389WC9-1NliaP6mpBU77ZkgW6v/view?usp=sharing)** in google colaboratory to reproduce the results. To access pretrained models visit **[link](https://drive.google.com/file/d/11U-MnbjkLQXK_z5R6RPsfSZWwmSPoj34/view?usp=sharing)**. The folder `predictions` has the generations from the pretrained models. 

**Note**: Pretrained models were trained with pytorch 0.3.1.

Generation and Evaluation of Simplifications 
--------
```
bash translate.sh
```
`translate.sh` is used for
- Generating simplifications of test dataset.
- Computing stand alone metrics such as Flesch readability ease score difference, Tree similarity and Document similarity metrics.
- Computing  SARI, BLEU and Word-diff metrics.

Acknowledgements
--------
Our code uses functions from https://github.com/artetxem/undreamt and https://github.com/senisioi/NeuralTextSimplification extensively.

If you use our system for academic research, please cite the following paper:
```
@inproceedings{surya-etal-2019-unsupervised,
    title = "Unsupervised Neural Text Simplification",
    author = "Surya, Sai  and
      Mishra, Abhijit  and
      Laha, Anirban  and
      Jain, Parag  and
      Sankaranarayanan, Karthik",
    booktitle = "Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics",
    month = jul,
    year = "2019",
    address = "Florence, Italy",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/P19-1198",
    doi = "10.18653/v1/P19-1198",
    pages = "2058--2068"
}
```
