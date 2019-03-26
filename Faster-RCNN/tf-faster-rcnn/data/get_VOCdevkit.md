##Beyond the demo: installation for training and testing models

1. Download the training, validation, test data and VOCdevkit

```
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCdevkit_08-Jun-2007.tar
```

2. xtract all of these tars into one directory named VOCdevkit

```
tar xvf VOCtrainval_06-Nov-2007.tar
tar xvf VOCtest_06-Nov-2007.tar
tar xvf VOCdevkit_08-Jun-2007.tar
```

3. t should have this basic structure

```
$VOCdevkit/                           # development kit
$VOCdevkit/VOCcode/                   # VOC utility code
$VOCdevkit/VOC2007                    # image sets, annotations, etc.
# ... and several other directories ...
```

4. Create symlinks for the PASCAL VOC dataset

```
cd $FRCN_ROOT/data
ln -s $VOCdevkit VOCdevkit2007
```

Using symlinks is a good idea because you will likely want to share the same PASCAL dataset installation between multiple projects.

5. [Optional] follow similar steps to get PASCAL VOC 2010 and 2012

6. [Optional] If you want to use COCO, please see some notes under data/README.md

7. Follow the next sections to download pre-trained ImageNet models