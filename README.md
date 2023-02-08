# Baseline model for the MedSecId paper

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7150451.svg)](https://doi.org/10.5281/zenodo.7150451)

This contains a basic baseline model for identifying section in clinical text
from the paper [A New Public Corpus for Clinical Section Identification:
MedSecId].  The purpose of this repository is to provide a means to reproduce
the results in the paper.  If you want to include this work in your own
projects, use the [mimicsid] package described in the [Inclusion in Your
Projects](#inclusion-in-your-projects) section, which was designed to be an
off-the-shelf package pip install.


## Reproducing the Results

Python 3.9.9 was used with the requirements in `src/requirements.txt` and
`src/requirements-mednlp.txt`.

To train and test the models, use the `run.sh` script by:

1. Copy the MIMIC-III `NOTEEVENTS.csv` file to the `corpus` directory.
2. Download the annotation set and uncompress it:
   1. `pushd corpus`
   2. `wget https://zenodo.org/record/7150451/files/section-id-annotations.zip`
   3. `unzip section-id-annotations.zip`
   4. `popd`
3. Remove the repo results: `rm -r results`
4. Create the Python environment (in `pyvirenv`): `./run.sh pyenv`
5. Install all Python libraries and models: `./run.sh pydep`
6. Create the features as mini-batches (takes a while): `./run.sh batch`
7. Test and train the models (takes a while): `./run.sh traintest`
8. Create the metrics used in the `./run paperresults`

At the end of this, there should be a `results` directory with:
* **`results/stats`**: the corpus statistics
* **`results/perf`**: the summary of the results and labels of the best model
* **`results/model`**: the models and model specific results


## Inclusion in Your Projects

The purpose of this repository is to reproduce the results in the paper.  If
you want to use the annotations and/or use the pretrained model, please refer
to the [mimicsid] repository.


## Data Analysis

The medical concept (CUI) plot given in the paper, and others are available [as
interactive 3D plots here](https://uic-nlp-lab.github.io/medsecid/index.html).


## Citation

If you use this project in your research please use the following BibTeX entry:

```bibtex
@inproceedings{landes-etal-2022-new,
    title = "A New Public Corpus for Clinical Section Identification: {M}ed{S}ec{I}d",
    author = "Landes, Paul  and
      Patel, Kunal  and
      Huang, Sean S.  and
      Webb, Adam  and
      Di Eugenio, Barbara  and
      Caragea, Cornelia",
    booktitle = "Proceedings of the 29th International Conference on Computational Linguistics",
    month = oct,
    year = "2022",
    address = "Gyeongju, Republic of Korea",
    publisher = "International Committee on Computational Linguistics",
    url = "https://aclanthology.org/2022.coling-1.326",
    pages = "3709--3721"
}
```

Also please cite the [Zensols Framework]:

```bibtex
@article{Landes_DiEugenio_Caragea_2021,
  title={DeepZensols: Deep Natural Language Processing Framework},
  url={http://arxiv.org/abs/2109.03383},
  note={arXiv: 2109.03383},
  journal={arXiv:2109.03383 [cs]},
  author={Landes, Paul and Di Eugenio, Barbara and Caragea, Cornelia},
  year={2021},
  month={Sep}
}
```


## License

[MIT License]

Copyright (c) 2022 Paul Landes


<!-- links -->

[MIT License]: https://opensource.org/licenses/MIT

[A New Public Corpus for Clinical Section Identification: MedSecId]: https://aclanthology.org/2022.coling-1.326.pdf
[mimicsid]: https://github.com/plandes/mimicsid
[Zensols Framework]: https://github.com/plandes/deepnlp
