# Deep Retrieval

This package contains the pretrained ResNet101 <del>model</del> and evaluation script for the method proposed in the following papers:

* *Deep Image Retrieval: Learning global representations for image search.* A. Gordo, J. Almazan, J. Revaud, and D. Larlus. In ECCV, 2016
* *End-to-end Learning of Deep Visual Representations for Image Retrieval.* A. Gordo, J. Almazan, J. Revaud, and D. Larlus. CoRR abs/1610.07940, 2016

### Dependencies:
 - Caffe
 - Region of Interest pooling layer (ROIPooling). This is the same layer used by fast RCNN and faster RCNN. A C++ implementation can be found in https://github.com/BVLC/caffe/pull/4163
 - L2-normalization layer (Normalize). Implemented in C++ in https://github.com/happynear/caffe-windows. As an alternative, we provide a python implementation of this layer that produces the same results, but is less efficient and does not implement backpropagation.


### Datasets
The evaluation script is prepared to work on the Oxford 5k and Paris 6k datasets. To set up the datasets:

```sh
mkdir datasets
cd datasets
```

**Evaluation script:**
```sh
mkdir evaluation
cd evaluation
wget http://www.robots.ox.ac.uk/~vgg/data/oxbuildings/compute_ap.cpp
sed -i '6i#include <cstdlib>' compute_ap.cpp # Add cstdlib, as some compilers will produce an error otherwise
g++ -o compute_ap compute_ap.cpp
cd ..
```

**Oxford:**
```sh
mkdir -p Oxford
cd Oxford
mkdir jpg lab
wget http://www.robots.ox.ac.uk/~vgg/data/oxbuildings/oxbuild_images.tgz
tar -xzf oxbuild_images.tgz -C jpg
wget http://www.robots.ox.ac.uk/~vgg/data/oxbuildings/gt_files_170407.tgz
tar -xzf gt_files_170407.tgz -C lab
cd ..
```

**Paris**
```sh
mkdir -p Paris
cd Paris
mkdir jpg lab tmp
# Images are in a different folder structure, need to move them around
wget http://www.robots.ox.ac.uk/~vgg/data/parisbuildings/paris_1.tgz
wget http://www.robots.ox.ac.uk/~vgg/data/parisbuildings/paris_2.tgz
tar -xzf paris_1.tgz -C tmp
tar -xzf paris_2.tgz -C tmp
find tmp -type f -exec mv {} jpg/ \;
rm -rf tmp
wget http://www.robots.ox.ac.uk/~vgg/data/parisbuildings/paris_120310.tgz
tar -xzf paris_120310.tgz -C lab
cd ..
cd ..
```

## Usage
```
$ python test.py

usage: test.py [-h] --gpu GPU --S S --L L --proto PROTO --weights WEIGHTS
               --dataset DATASET --dataset_name DATASET_NAME --eval_binary
               EVAL_BINARY --temp_dir TEMP_DIR [--multires] [--aqe AQE]
               [--dbe DBE]

G: gpu id
S: size to resize the largest side of the images to. The model is trained with S=800, but different values may work better depending on the task.
L: number of levels of the rigid grid. Model was trained with L=2, but different levels (e.g. L=1 or L=3) may work better on other tasks.
PROTO: path to the prototxt. There are two prototxts included.
  deploy_resnet101.prototxt relies on caffe being compiled with the normalization layer.
  deploy_resnet101_normpython.prototxt does not have that requirement as it relies on the python implementation, but it may be slower as it is done on the cpu and does not implement backpropagation.
WEIGHTS: path to the caffemodel
DATASET: path to the dataset, for Oxford and Paris it is the directory that contains the jpg and lab folders.
DATASET_NAME: either Oxford or Paris
EVAL_BINARY: path to the compute_ap binary provided with Oxford and Paris used to compute the ap scores
TEMP_DIR: a temporary directory to store features and scores
```

Note that this model does not implement the region proposal network.

## Examples
Adjust paths as necessary:

**Rigid grid, no multiresolution, no query expansion or database side feature augmentation:**

```sh
python test.py --gpu 0 --S 800 --L 2 --proto deploy_resnet101_normpython.prototxt --weights model.caffemodel --dataset datasets/Oxford --eval_binary datasets/evaluation/compute_ap --temp_dir tmp --dataset_name Oxford
```
Expected accuracy: `84.09`

```sh
python test.py --gpu 0 --S 800 --L 2 --proto deploy_resnet101_normpython.prototxt --weights model.caffemodel --dataset datasets/Paris --eval_binary datasets/evaluation/compute_ap --temp_dir tmp --dataset_name Paris
```
Expected accuracy: `93.57`

**Rigid grid, multiresolution, no query expansion or database side feature augmentation:**
```sh
python test.py --gpu 0 --S 800 --L 2 --proto deploy_resnet101_normpython.prototxt --weights model.caffemodel --dataset datasets/Oxford --eval_binary datasets/evaluation/compute_ap --temp_dir tmp --dataset_name Oxford --multires
```
Expected accuracy: `86.07`

```sh
python test.py --gpu 0 --S 800 --L 2 --proto deploy_resnet101_normpython.prototxt --weights model.caffemodel --dataset datasets/Paris --eval_binary datasets/evaluation/compute_ap --temp_dir tmp --dataset_name Paris --multires
```
Expected accuracy: `94.53`

**Rigid grid, multiresolution, query expansion (k=1) and database side feature augmentation (k=20):**
```sh
python test.py --gpu 0 --S 800 --L 2 --proto deploy_resnet101_normpython.prototxt --weights model.caffemodel --dataset datasets/Oxford --eval_binary datasets/evaluation/compute_ap --temp_dir tmp --dataset_name Oxford –multires --aqe 1 --dbe 20
```
Expected accuracy: `94.68`

```sh
python test.py --gpu 0 --S 800 --L 2 --proto deploy_resnet101_normpython.prototxt --weights model.caffemodel --dataset datasets/Paris --eval_binary datasets/evaluation/compute_ap --temp_dir tmp --dataset_name Paris –multires --aqe 1 --dbe 20
```
Expected accuracy: `96.58`

### Citation

If you use these models in your research, please cite:

```
@inproceedings{Gordo2016a,
      title={Deep Image Retrieval: Learning global representations for image search},
      author={Albert Gordo and Jon Almazan and Jerome Revaud and Diane Larlus},
      booktitke={ECCV},
      year={2016}
}   
@article{Gordo2016b,
      title={End-to-end Learning of Deep Visual Representations for Image Retrieval}
      author={Albert Gordo and Jon Almazan and Jerome Revaud and Diane Larlus},
      journal={CoRR abs/1610.07940},
      year={2016}
}
```

Please see `LICENSE.txt` for the license information.
