# tfswin

Keras (TensorFlow v2) reimplementation of **Swin Transformer** and **Swin Transformer V2** models with Unet for Image Segmentation tasks.

+ Based on [Swin-Unet: Unet-like Pure Transformer for
Medical Image Segmentation](https://arxiv.org/pdf/2105.05537.pdf).
+ Supports variable-shape inference for downstream tasks.
+ Contains pretrained weights converted from official ones.

## Examples

Default usage (without preprocessing):

```python
from tfswin import SwinTransformerTiny224  # + 5 other variants and input preprocessing

# or 
# from tfswin import SwinTransformerV2Tiny256  # + 5 other variants and input preprocessing

# Important!! Input and output shapes must be provided for weight and layer calculations
model = SwinTransformerTiny224(input_shape=(224, 224, 3), output_shape = (224,224,1))  # by default will download imagenet[21k]-pretrained weights
model.compile(...)
model.fit(...)
```

Custom Segmentation (with preprocessing):

```python
from keras import layers, models
from tfswin import SwinTransformerTiny224, preprocess_input

inputs = layers.Input(shape=(224, 224, 3), dtype='uint8')
outputs = layers.Lambda(preprocess_input)(inputs)
outputs = SwinTransformerTiny224(input_shape=(224, 224, 3), output_shape = (224,224,1))(outputs)
outputs = layers.Dense(100, activation='softmax')(outputs)

model = models.Model(inputs=inputs, outputs=outputs)
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
