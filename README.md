# The code for "A Flexible Framework for Local Face Attribute Transfer"

## Introduction
In the context of social media, a large number of headshot photos are taken in everyday life for sharing and entertainment. Unfortunately, besides laborious editing and modification, creating a visually compelling photographic masterpiece for sharing on social networking sites requires the advanced professional skills, which are difficult for ordinary Internet users to master. Though there are many algorithms automatically transferring the style from one image to another in a global way, these algorithms fail to respect the semantics of the scene, and are not flexible enough to allow users to merely transfer the attributes of one or two face organs while maintaining other attributes unchanged. To mitigate this problem, we developed a novel framework for semantically-meaningful local face attribute transfer, which can flexibly transfer the local attribute of a face organ from the reference image to a semantically equivalent organ in the input image. Our method involves warping the reference photo to match the shape, pose, location, and expression of the input image by thin plate spline. After that, we take the fusion of the warped reference image and input image as the initialization image for neural style transfer algorithm. The experimental results are provided to demonstrate the efficacy of the proposed framework, reveal its superiority over other state-of-the-art alternatives.

## Dependnecy
python 3.5, pyTorch >= 1.4.0 (from https://pytorch.org/), numpy, Pillow, dlib, cv2, Scikit.
## Usage

1. Prepare your own input image, reference image, and the semantic map of them, and change the data path in the "run_code.py". In this repository, the input and refernce image are "in11.png" and "tar11.png", respectively, and the semantic map of them are in11_seg.png and tar11_seg.png, respectively. 

2. Starting excuting using the following command

```python run_code.py```


## Results
![Reesuly](img/exp.png)
![Reesuly](img/ourf.png)
More Results can be found in our website: https://forawardstar.github.io/EDIT-Project-Page/

