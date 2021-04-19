# FS-KTN.PyTorch

[![LICENSE](https://img.shields.io/badge/license-MIT-green)](https://github.com/CSer-Tang-hao/FS-KTN/blob/master/LICENSE)
[![Python](https://img.shields.io/badge/python-3.6-blue.svg)](https://www.python.org/)
![PyTorch](https://img.shields.io/badge/pytorch-1.2.0-%237732a8)

A simple PyTorch implementation of KTN ( Knowledge Transfer Network ) for FS ( Few-Shot Image Classification ). 

(_Peng et al._, [Few-Shot Image Recognition with Knowledge Transfer](http://openaccess.thecvf.com/content_ICCV_2019/papers/Peng_Few-Shot_Image_Recognition_With_Knowledge_Transfer_ICCV_2019_paper.pdf), ICCV2019)

**NOTICE: This is NOT an official implementation by authors of FS-KTN. This version shown here is indicative only, please be subject to the  official implementation which may be available soon.** 

## Performance
This code is implemented on PyTorch and the experiments were done on a 1080Ti GPU ( batch_size = 64 ).
So you may need to install
* Python==3.x
* torch==1.2.0 or above
* torchvision==0.4.0
* torchnet==0.0.4
* tqdm


[comment]: <> (|Model|Dataset|Backbone|1-Shot &#40;Our&#41;|5-Shot &#40;Our&#41;|)

[comment]: <> (|:-----:|:-----:|:----:|:--------------:|:--------------:|)

[comment]: <> (|Vis.|[MiniImagenet]&#40;https://mega.nz/#!rx0wGQyS!96sFlAr6yyv-9QQPCm5OBFbOm4XSD0t-HlmGaT5GaiE&#41;|Conv64|54.34 +- 0.78% |71.40 +- 0.63%|)

[comment]: <> (|Vis.+Kno.|[MiniImagenet]&#40;https://mega.nz/#!rx0wGQyS!96sFlAr6yyv-9QQPCm5OBFbOm4XSD0t-HlmGaT5GaiE&#41;|Conv64|63.83 +- 0.77%|73.63 +- 0.62%|)



## Usage

### Dataset Directory

* [MiniImageNet](https://mega.nz/#!rx0wGQyS!96sFlAr6yyv-9QQPCm5OBFbOm4XSD0t-HlmGaT5GaiE)

  ```
  -/Datasets/MiniImagenet/
                  └─── miniImageNet_category_split_train_phase_train.pickle
                  └─── miniImageNet_category_split_train_phase_val.pickle
                  └─── miniImageNet_category_split_train_phase_test.pickle
                  └─── miniImageNet_category_split_val.pickle
                  └─── miniImageNet_category_split_test.pickle
  ```
  
### Run

This repo contains FS-KTN with feature extractors using Conv64 / Conv128 in PyTorch form, see ```./models/Conv_model.py```. 

1. ``` git clone``` this repo.
2. Prepare data files in ```./Datasets/MiniImagenet/```.
3. **Set configurations** in ```Train_only_Vis.py``` ( Training / Valing Config, Model Config, Dataset/Path Config):
4. ```$ python Train_only_Vis.py --network Conv64/128``` for training. ( tqdm package is required. Other logs are written in ```<save_dir>/train_log.txt```).
5. ```$ python Test_only_Vis.py --network Conv64/128 --test_nExemplars 1/5```  for testing only using Vision-based Classifier (Baseline).
6. Download the following [Knowledge Graph for MiniImageNet](https://drive.google.com/file/d/1o7URkid8r9fhmySUbwbywSq-sKnEy6jk/view?usp=sharing) into `./mini-graph` folder. ( Now we only provide generated **Knowledge Graph** for MiniImageNet，please refer to [DGP](https://github.com/yinboc/DGP/tree/master/materials)  for more details about it ).
7. ```$ python Test_Vis_Kno.py --network Conv64 --test_nExemplars 1/5```  for testing using Vision-Knowledge Classifier. 

## Citation
If this work is useful in your research, please cite 

```
@InProceedings{Peng_2019_ICCV,
author = {Peng, Zhimao and Li, Zechao and Zhang, Junge and Li, Yan and Qi, Guo-Jun and Tang, Jinhui},
title = {Few-Shot Image Recognition With Knowledge Transfer},
booktitle = {The IEEE International Conference on Computer Vision (ICCV)},
month = {October},
year = {2019}
}
```

## References
This implementation builds upon several open-source codes. Specifically, we have modified and integrated the following codes into this repository:

*  [FewShotWithoutForgetting](https://github.com/gidariss/FewShotWithoutForgetting) 
*  [DGP](https://github.com/cyvius96/DGP) 

