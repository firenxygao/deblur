# Dynamic Scene Deblurring with Parameter Selective Sharing and Nested Skip Connections
by Hongyun Gao, Xin Tao, Xiaoyong Shen, Jiaya Jia. Please refer to the [paper](http://jiaya.me/papers/deblur_cvpr19.pdf) for the details.

### Some results on the GoPro testing dataset by our model trained on default dataset
<img src="./imgs/comp_gopro.png" width="100%">
From the top to bottom, input images, results of [1], [2], [3] and ours are shown.

### Some results on more blurred images by our model trained on mixing datasets
<img src="./imgs/comp_real.png" width="100%">
The first column is the input images. The second column is generated by [2]. The third column is produced by [3]. The fourth column is our results.

## Prerequisites
- Python2.7 or Python3.6
- Opencv3.4
- Numpy
- Tensorflow 1.7 with NVIDIA GPU or CPU (cpu testing is very slow)

## Installation
Clone this repository to your PC. 

```bash
git clone https://github.com/firenxygao/deblur.git
cd deblur
```

## Testing

If GPU is available, you can use `--gpu` argument and add the gpu id to enable GPU computation. Otherwise, the code will use CPU for computation. 

```bash
python run_model.py --gpu=0
```

We provide 2 models for testing. The first model is trained on default data released by the paper ''Deep Multi-scale Convolutional Neural Network for Dynamic Scene Deblurring''. The second model is trained by mixing default data with our own generated data, which shows better performance than the first model. You can use `--model` argument to choose between `default` or `alldata`.
Our generated data can be downloaded by the links. [Dataset](https://drive.google.com/file/d/18__gfWalgOHA2vVUPGiCjLaR6UaRMSLv/view?usp=sharing).

```bash
python run_model.py --model=default
```

You can test one single image or a folder of images by using `--input_path` argument. If you test images in `testing_imgs`, the output images will be saved into `testing_imgs_res`. If you test one single image `testing_img.jpg`, the result will be named `testing_img_res.jpg`.

```bash
python run_model.py --input_path=testing_imgs
```

To test the model, the height and width of the input tensor should be pre-defined as `--max_height` and `--max_width`. Our network requires the height and width to be multiples of `16` and the dimension should be assigned to the maximum size to accommodate all the images. 

In our implementation, we first check the image dimension. If the height is larger than the width, we transpose the image such that the width is larger than the height. Then we check whether image can be fit into the placeholder pre-defined by `max_height` and `max_width`. Otherwise, the images will be downsampled by the largest scale factor to 
be fed into the placeholder. And results will be upsampled to the original size.

According to our experience, a 720\*1280 image will take 9GB memory. Users can adjust `max_height` and `max_width` to satisfy memory conditions.

```bash
python run_model.py --max_height=720 --max_width=1280
```

## Citation
If you find our released models or dataset useful, please consider citing: 

```bibtex
@inproceedings{gao2019dynamic,
  title={Dynamic scene deblurring with parameter selective sharing and nested skip connections},
  author={Gao, Hongyun and Tao, Xin and Shen, Xiaoyong and Jia, Jiaya},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={3848--3856},
  year={2019}
}
```

## Reference
[1] `Gong et al.` D. Gong, J. Yang, L. Liu, Y. Zhang, I. D. Reid, C. Shen, A. Van Den Hengel, and Q. Shi. *From motion blur to motion flow: A deep learning solution for removing heterogeneous motion blur.* In CVPR, pages 2319–2328, 2017.

[2] `Nah et al.` S. Nah, T. H. Kim, and K. M. Lee. *Deep multi-scale convolutional neural network for dynamic scene deblurring.* In CVPR, pages 3883–3891, 2017.

[3] `Tao et al.` X. Tao, H. Gao, X. Shen, and J. Jia. *Scale-recurrent network for deep image deblurring.* In CVPR, pages 8174–8182, 2018.