# Neural Network-Based Image Recolourisation Method

## Background Setup and Preprocessing

All training images are cropped to 256 × 256. For each image, between 20 and 200 points are randomly sampled to generate the training data.

```python
CROP_SIZE = 256
point_cfg = PointSamplerConfig(
    mode="random",
    min_points=20,
    max_points=200,
    radius=1,
)
```

## LAB Colour Encoding

We adopt the LAB colour space because, in previous parameter optimisation experiments, we observed that the machine loss function was not fully aligned with the perceptual loss of the human visual system. The three channels in LAB are: L, a, and b.

See: https://en.wikipedia.org/wiki/CIELAB_color_space

However, it is unclear whether LAB encoding can be directly applied in the previous parameter optimisation framework. Unlike YCbCr, LAB is not a simple linear basis transformation, and it may destroy the convexity of the original optimisation problem.

## Neural Network: Input and Output

$$
\mathcal{F}_\theta: \mathbb{R}^{4 \times H \times W} \rightarrow \mathbb{R}^{2 \times H \times W}
$$

**Input:**

$$
(B, 4, H, W)
$$

- $B$: batch size
- $H$: height
- $W$: width

The four channels are: $(L, a, b, \text{mask})$, where the mask is the hint mask: it equals 1 if the location is a hint point, and 0 otherwise.

**Output:**

$$
(B, 2, H, W)
$$

The two output channels are:

$$
(\hat{a}(x), \hat{b}(x)).
$$

## U-Net Architecture

In one sentence, U-Net consists of an encoder, a decoder, and skip connections.

Assume the input size is $H \times W$.

We design four downsampling stages:

$$
H \rightarrow \frac{H}{2} \rightarrow \frac{H}{4} \rightarrow \frac{H}{8} \rightarrow \frac{H}{16}.
$$

Channel progression, namely the number of convolutional filters, is:

$$
4 \rightarrow 64 \rightarrow 128 \rightarrow 256 \rightarrow 512 \rightarrow 1024.
$$

Four upsampling stages are then used:

$$
\frac{H}{16} \rightarrow \frac{H}{8} \rightarrow \frac{H}{4} \rightarrow \frac{H}{2} \rightarrow H.
$$

The corresponding channel progression is:

$$
1024 \rightarrow 512 \rightarrow 256 \rightarrow 128 \rightarrow 64 \rightarrow 2.
$$

### Upsampling Implementation

Suppose we are at the bottom of the network:

$$
(B, 1024, H/16, W/16).
$$

**Step 1:**

Upsample the spatial resolution by a factor of 2, using bilinear upsampling or `ConvTranspose2d`, and halve the number of channels via convolution:

$$
(B, 512, H/8, W/8).
$$

**Step 2:**

Apply a skip connection by concatenating the feature map from the corresponding encoder layer:

$$
(B, 512, H/8, W/8) + (B, 512, H/8, W/8) = (B, 1024, H/8, W/8).
$$

**Step 3:**

Apply convolution to reduce the number of channels:

$$
(B, 512, H/8, W/8).
$$

### Intuitive Interpretation of U-Net

Although human intuition is not always reliable for understanding neural networks:

- Deeper layers provide colour context, since colour consistency is influenced by neighbouring pixels and U-Net captures neighbourhood structure via convolution.
- Same-level skip connections provide texture details, which are already available in the original input and do not need to be relearned from scratch.

## Loss Function

$$
\mathcal{L}=\mathcal{L}_{\text{global}}+\lambda_{\text{hint}}\mathcal{L}_{\text{hint}}+\lambda_{\mathrm{tv}}\mathcal{L}_{\mathrm{tv}}.
$$

where

$$
\mathcal{L}_{\text{global}}=\frac{1}{H W}\sum_x\left(|\hat{a}(x)-a(x)|+|\hat{b}(x)-b(x)|\right),
$$

$$
\mathcal{L}_{\text{hint}}=\frac{1}{|\Omega|}\sum_{x \in \Omega} m(x)\left(|\hat{a}(x)-a(x)|+|\hat{b}(x)-b(x)|\right),
$$

and

$$
\mathcal{L}_{\mathrm{tv}}=\frac{1}{|\Omega|}\sum_{x \in \Omega}\left(\left|\partial_x \hat{a}(x)\right|+\left|\partial_y \hat{a}(x)\right|+\left|\partial_x \hat{b}(x)\right|+\left|\partial_y \hat{b}(x)\right|\right).
$$

The TV term is used to suppress high-frequency noise. In practice, we set:

- $\lambda_{\text{hint}} = 0$, meaning that no additional penalty is imposed on hint points;
- $\lambda_{\text{tv}} = 0.01$, meaning that a small penalty is imposed on high-frequency components.

## U-Net-Based Image Reconstruction

All training images are cropped to 256 × 256 due to computational constraints. During reconstruction, we only require that the input height and width be divisible by 16; they do not need to be equal.

In the implementation, we crop the height and width to the largest multiples of 16 that do not exceed the original dimensions.
