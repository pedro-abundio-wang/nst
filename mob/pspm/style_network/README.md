# Style Network

The neural network is a combination of Gatys' [A Neural Algorithm of Artistic Style](https://arxiv.org/abs/1508.06576), Johnson's [Perceptual Losses for Real-Time Style Transfer and Super-Resolution](http://cs.stanford.edu/people/jcjohns/eccv16/), and Ulyanov's [Instance Normalization](https://arxiv.org/abs/1607.08022). 

## Image Stylization :art:

Added styles from various paintings to a photo of Chicago. Check the ./images/results folder to see more images.

<div align='center'>
<img src = 'images/content/chicago.jpg' height="200px">
</div>
<div align = 'center'>
<a href = 'images/style/wave.jpg'><img src = 'images/thumbs/wave.jpg' height = '200px'></a>
<img src = 'images/results/wave.jpg' height = '200px'>
<img src = 'images/results/africa.jpg' height = '200px'>
<a href = 'images/style/africa.jpg'><img src = 'images/thumbs/africa.jpg' height = '200px'></a>
<br>
<a href = 'images/style/aquarelle.jpg'><img src = 'images/thumbs/aquarelle.jpg' height = '200px'></a>
<img src = 'images/results/aquarelle.jpg' height = '200px'>
<img src = 'images/results/shipwreck.jpg' height = '200px'>
<a href = 'images/style/the_shipwreck_of_the_minotaur.jpg'><img src = 'images/thumbs/the_shipwreck_of_the_minotaur.jpg' height = '200px'></a>
<br>
<a href = 'images/style/starry_night.jpg'><img src = 'images/thumbs/starry_night.jpg' height = '200px'></a>
<img src = 'images/results/starry_night.jpg' height = '200px'>
<img src = 'images/results/hampson.jpg' height = '200px'>
<a href = 'images/style/hampson.jpg'><img src = 'images/thumbs/hampson.jpg' height = '200px'></a>
<br>
<a href = 'images/style/chinese_style.jpg'><img src = 'images/thumbs/chinese_style.jpg' height = '200px'></a>
<img src = 'images/results/chinese_style.jpg' height = '200px'>
<img src = 'images/results/udnie.jpg' height = '200px'>
<a href = 'images/style/udnie.jpg'><img src = 'images/thumbs/udnie.jpg' height = '200px'></a>
</div>
<p align = 'center'>
All the models were trained on the same default settings.
</p>

## Implementation Details

- The **transformation network** is roughly the same as described in Johnson, except that batch normalization is replaced with Ulyanov's instance normalization, and the scaling/offset of the output `tanh` layer is slightly different (for better convergence), also use [Resize-convolution layer](https://distill.pub/2016/deconv-checkerboard/) to replace the regular transposed convolution for better upsampling (to avoid checkerboard artifacts)
- The **loss network** which is similar to the one described in Gatys, using VGG19 instead of VGG16 and typically using "shallower" layers than in Johnson's implementation, for larger scale style features in transformation (e.g. use `relu1_1` rather than `relu1_2`).

### Training Style Transfer Networks

Use `main.py` to train a style transfer network.

Example usage:

    python main.py train \
      --style ./path/to/style/image.jpg \
      --dataset ./path/to/mscoco \
      --weights ./path/to/model/weights \

### Evaluating Style Transfer Networks

Use `main.py` to evaluate a style transfer network.

Example usage:

    python main.py evaluate \
      --weights ./path/to/model/weights \
      --content ./path/to/content/image.jpg(video.mp4) \
      --result ./path/to/save/results/image.jpg
