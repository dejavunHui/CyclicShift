
# CylicShift

This is Pytorch implementation of CyclicShift in the paper: *CyclicShift: A Data Augmentation Method For Enriching Data Patterns*

# Main Requirements
* torch == 1.0.1
* torchvision == 0.2.0
* Python 3
# Training Examples

## CyclicShift
### CIFAR100
```python
python cifar_cyclicshift.py -a resnet --dataset cifar100 --depth 56 --epochs 300 --schedule 150 225 --gamma 0.1 --wd 1e-4 --checkpoint checkpoint/cifar100/resnet-56-cyclicshift-1 --p 0.5
python cifar_cyclicshift.py -a resnet --dataset cifar100 --depth 110 --epochs 300 --schedule 150 225 --gamma 0.1 --wd 1e-4 --checkpoint checkpoint/cifar100/resnet-110-cyclicshift-1 --p 0.5
python cifar_cyclicshift.py -a wrn --dataset cifar100 --depth 28 --depth 28 --widen-factor 10 --drop 0.3 --epochs 200 --schedule 60 120 160 --wd 5e-4 --gamma 0.1 --checkpoint checkpoint/cifar100/WRN-28-10-drop-cyclicshift-1 --p 0.5
```
### CIFAR10
```python
python cifar_cyclicshift.py -a resnet --dataset cifar10 --depth 56 --epochs 300 --schedule 150 225 --gamma 0.1 --wd 1e-4 --checkpoint checkpoint/cifar10/resnet-56-cyclicshift-1 --p 0.5
python cifar_cyclicshift.py -a resnet --dataset cifar10 --depth 110 --epochs 300 --schedule 150 225 --gamma 0.1 --wd 1e-4 --checkpoint checkpoint/cifar10/resnet-110-cyclicshift-1 --p 0.5
python cifar_cyclicshift.py -a wrn --dataset cifar10 --depth 28 --depth 28 --widen-factor 10 --drop 0.3 --epochs 200 --schedule 60 120 160 --wd 5e-4 --gamma 0.1 --checkpoint checkpoint/cifar10/WRN-28-10-drop-cyclicshift-1 --p 0.5
```
### Tiny ImageNet
```python
python tiny_cyclicshift.py -a resnet --dataset ./tiny --depth 110 --epochs 300 --schedule 150 225 --gamma 0.1 --wd 1e-4 --checkpoint checkpoint/tiny/resnet-110-cyclicshift-1 --p 0.5
```
### ImageNet
```python
python imagenet_cyclicshift.py -a resnet50 --epochs 300 --schedule 75 150 225 --gamma 0.1 -c checkpoints/imagenet/resnet50-cyclicshift --p 0.5 --gpu-id 0,1,2,3
```
## CyclicMix
### CIFAR100
```python
python cifar_cyclicshift.py -a resnet --dataset cifar100 --depth 56 --epochs 300 --schedule 150 225 --gamma 0.1 --wd 1e-4 --checkpoint checkpoint/cifar100/resnet-56-cyclicshift-1 --p 0.5 --beta 1 --cutmix_prob 0.8
python cifar_cyclicshift.py -a resnet --dataset cifar100 --depth 110 --epochs 300 --schedule 150 225 --gamma 0.1 --wd 1e-4 --checkpoint checkpoint/cifar100/resnet-110-cyclicshift-1 --p 0.5 --beta 1 --cutmix_prob 0.8
python cifar_cyclicshift.py -a wrn --dataset cifar100 --depth 28 --depth 28 --widen-factor 10 --drop 0.3 --epochs 200 --schedule 60 120 160 --wd 5e-4 --gamma 0.1 --checkpoint checkpoint/cifar100/WRN-28-10-drop-cyclicshift-1 --p 0.5 --beta 1 --cutmix_prob 0.8
```
### Tiny ImageNet
```python
python tiny_cyclicshift.py -a resnet --dataset ./tiny --depth 110 --epochs 300 --schedule 150 225 --gamma 0.1 --wd 1e-4 --checkpoint checkpoint/tiny/resnet-110-cyclicshift-1 --p 0.5 --beta 1 --cutmix_prob 0.8
```
### ImageNet
```python
python imagenet_cyclicshift.py -a resnet50 --epochs 300 --schedule 75 150 225 --gamma 0.1 -c checkpoints/imagenet/resnet50-cyclicshift --p 0.5 --beta 1 --cutmix_prob 0.8 --gpu-id 0,1,2,3 
```