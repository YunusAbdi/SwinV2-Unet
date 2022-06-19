# tfswin

Keras (TensorFlow v2) reimplementation of **Swin Transformer** and **Swin Transformer V2** models with Unet for Image Segmentation tasks.

+ Based on [Swin-Unet: Unet-like Pure Transformer for
Medical Image Segmentation](https://arxiv.org/pdf/2105.05537.pdf).
+ Supports variable-shape inference for downstream tasks.
+ Contains pretrained weights converted from official ones.

## Examples

Default usage (with preprocessing):
Please note that by default the model takes images in the uint8 format with pixel values between 0-255
The preprocess layer preprocesses the images using imagenet preprocessing. If you want to use imagenet weights please use this preprocessing procedure as Transformers are sensitive in this regard!

```python
from tfswin import SwinTransformerTiny224  # + 5 other variants

# or 
# from tfswin import SwinTransformerV2Tiny256  # + 5 other variants

# Important!! Input and output shapes must be provided for weights and layer calculations
model = SwinTransformerTiny224(input_shape=(224, 224, 3), output_shape = (224,224,1))  # by default will download imagenet[21k]-pretrained weights and preprocess input if argument preprocess is not given!
model.compile(...)
model.fit(...)
```

Custom Segmentation (without preprocessing):

```python
from keras import layers, models
from tfswin import SwinTransformerTiny224


model = SwinTransformerTiny224(input_shape=(224, 224, 3), output_shape = (224,224,1), preprocess=False)


model.compile(...)
model.fit(...)
```

## Differences

Code simplification:

- Pretrain input height and width are always equal
- Patch height and width are always equal
- All input shapes automatically evaluated (not passed through a constructor like in PyTorch)
- Downsampling have been moved out from basic layer to simplify feature extraction in downstream tasks.
- SwinV1 accepts all image sizes but other parameters might break when using imagenet weights!
- SwinV2 is compatable with all image sizes and parameters (window_size, num_heads, etc.)

Performance improvements:

- Layer normalization epsilon fixed at `1.001e-5`, inputs are casted to `float32` to use fused op implementation.
- Some layers have been refactored to use faster TF operations.
- A lot of reshapes have been removed. Most of the time internal representation is 4D-tensor.
- Attention mask and relative index estimations moved to basic layer level.

## Variable shapes

When using Swin models with input shapes different from pretraining one, try to make height and width to be multiple
of `32 * window_size`. Otherwise a lot of tensors will be padded, resulting in speed and (possibly) quality degradation.



## Citation

```
@inproceedings{liu2021Swin,
  title={Swin Transformer: Hierarchical Vision Transformer using Shifted Windows},
  author={Liu, Ze and Lin, Yutong and Cao, Yue and Hu, Han and Wei, Yixuan and Zhang, Zheng and Lin, Stephen and Guo, Baining},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  year={2021}
}
```

```
@inproceedings{liu2021swinv2,
  title={Swin Transformer V2: Scaling Up Capacity and Resolution}, 
  author={Ze Liu and Han Hu and Yutong Lin and Zhuliang Yao and Zhenda Xie and Yixuan Wei and Jia Ning and Yue Cao and Zheng Zhang and Li Dong and Furu Wei and Baining Guo},
  booktitle={International Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2022}
}
```

```
@inproceedings{cao2021swin-unet,
  title={Swin-Unet: Unet-like Pure Transformer for Medical Image Segmentation}, 
  author={Hu Cao, Yueyue Wang, Joy Chen, Dongsheng Jiang, Xiaopeng Zhang, Qi Tian, Manning Wang},
  booktitle={arXiv:2105.05537v1 [eess.IV]},
  year={2021}
}
```
