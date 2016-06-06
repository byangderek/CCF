# README #

The codes are for the ICCV2015 paper ["Convolutional Channel Features"](http://arxiv.org/abs/1504.07339).

The codes include training and testing of a pedestrian detector on Caltech. We also provide our trained model for repreduction of the results in the paper.

The codes are written in Matlab, dependent on Caffe toolkit and Piotr's Computer Vision Matlab Toolbox. Codes are tested on Linux 12.04.3 LTS with 128GB memory.

### Preparation ###

* Make the provided Caffe with matCaffe interface
* Download [VGG-16 CaffeModel](https://gist.github.com/ksimonyan/211839e770f7b538e2d8#file-readme-md) to `./data/CaffeNets/`
* Download [Caltech Pedestrian Dataset](http://www.vision.caltech.edu/Image_Datasets/CaltechPedestrians/) and set it up properly with codes in `./data/code3.2.1`

### Demo for pedestrian detection ###

* Run `./runDetect.m`, and detection results will be saved as `allBBs.mat`

### Train a pedestrian detector ###

* Run modified `./toolbox-master/detector/acfDemoCal.m` to train an ACF detector and save the collected bbs for training
* Run `./getFeat_train.m` to extract features of training data
* Run `./trainModel.m` to train the boosting decision tree model
* Run `./runDetect.m` using your trained model
* (Evaluation) Please refer to `./toolbox-master/detector/acfTest.m` and `./data/code3.2.1/dbEval.m`

### Power law ###

* Power law can accelerate the detection time by a large margin with unnoticeable performance decrease. To our knowledge, it hold for CCF on AFW face data set but doesn't hold on Caltech pedestrian data set. (see the paper for more details)
* You can check whether the power law holds for specific feature type on specific data set using codes in `./power_law`. Just run `getMean.m` and `getLambda.m` consecutively.

### Reference ###

* If you use our codes or models in your research, we are grateful if you cite the paper
```
@inproceedings{binyang15ccf,
  Author    = {Bin Yang and
               Junjie Yan and
               Zhen Lei and
               Stan Z. Li},
  Title     = {Convolutional Channel Features},
  Booktitle = {Proceedings of the IEEE International Conference
               on Computer Vision (ICCV)},
  Year      = {2015}
}
```

### Acknowledgement ###

Great gratitude is presented to

* Piotr Dollar's toolbox
* Caffe team
* VGG team
* NVIDIA Corporation

--
* by Bin Yang
* 6.6.2016