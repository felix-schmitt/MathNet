[![CC BY 4.0][cc-by-shield]][cc-by]

# MathNet: A Data-Centric Approach for Printed Mathematical Expression Recognition

Printed mathematical expression recognition (MER) models are usually trained and tested using LaTeX-generated mathematical expressions (MEs) as input and the LaTeX source code as ground truth. As the same ME can be generated by various different LaTeX source codes, this leads to unwanted variations in the ground truth data that bias test performance results and hinder efficient learning. In addition, the use of only one font to generate the MEs heavily limits the generalization of the reported results to realistic scenarios. We propose a data-centric approach to overcome this problem, and present convincing experimental results: Our main contribution is an enhanced LaTeX normalization to map any LaTeX ME to a canonical form. Based on this process, we developed an improved version of the benchmark dataset im2latex-100k, featuring $30$ fonts instead of one. Second, we introduce the real-world dataset realFormula, with MEs extracted from papers. Third, we developed a MER model, MathNet, based on a convolutional vision transformer, with superior results on all four test sets (im2latex-100k, im2latexv2, realFormula, and InftyMDB-1), outperforming the previous state of the art by up to 88.3%.

## im2latexv2

You can download the im2latexv2 dataset from Zenodo ([Part 1](https://zenodo.org/records/11230382): 10.5281/zenodo.11230382, [Part 2](https://zenodo.org/records/11296280): 10.5281/zenodo.11296280)

## realFormula

You can download the realFormula dataset from [Zenodo](https://zenodo.org/records/11296815) (doi: 10.5281/zenodo.11296815)

## Inference using a Pretrained Model

The naming of the vit-pytorch models has been changed (a warning about missing keys is issued when loading the model). 
To run the model from dropbox, you have to install vit-pytorch==0.40.2.
Get the model from [dropbox](https://www.dropbox.com/scl/fo/xefqjzfd8szj6ra3f02hc/APcpMTDcRdp6sf3ZRCYkyOA?rlkey=5z9xlok8zwk87htnv57b7n9i7&st=umuhu5i3&dl=0) and save it in the trainedModels folder. 
You can inference an image with the [inference.py](inference.py) script. The image should have ideally a solution of 600 DPI by a font size of 12. 

## Train a New Model
Adapt the [config file](configs/im2latexv2-cvt.yaml) and the [dataset config file](configs/datasets/dataset-im2latexv2.yaml) if required.


## License
This work is licensed under a
[Creative Commons Attribution 4.0 International License][cc-by].

[![CC BY 4.0][cc-by-image]][cc-by]

[cc-by]: http://creativecommons.org/licenses/by/4.0/
[cc-by-image]: https://i.creativecommons.org/l/by/4.0/88x31.png
[cc-by-shield]: https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg
## Citation

### MathNet: A Data-Centric Approach for Printed Mathematical Expression Recognition

Felix M. Schmitt-Koopmann, Elaine M. Huang, Hans-Peter Hutter, Thilo Stadelmann, Alireza Darvishy

```
@ARTICLE{9869643,
    author={Schmitt-Koopmann, Felix M. and Huang, Elaine M. and Hutter, Hans-Peter and 
    Stadelmann, Thilo and Darvishy, Alireza},  
    journal={IEEE Access},   
    title={MathNet: A Data-Centric Approach for Printed Mathematical Expression Recognition},   
    year={2024},   
    doi={10.1109/ACCESS.2024.3404834}}
```