# Context and Introduction

This work aims to improve mammograms quality, using ML methods. It has been done after reading the following article: "Deep learning-based denoising of mammographic images using physics-driven data augmentation", written by the Pattern recognition lab of Friedrich-Alexander-Universität, together with Siemens Healthcare's mammography R\&D. 

This model has been trained on the well known MIAS (Mammographic Image Analysis Society) dataset.

# The problem

Breast cancer is the most commonly diagnosed cancer among women: according to the WHO, it ranked first in 157 of 185 countries in 2022, and caused an estimated 670,000 deaths worldwide that year.

This mortality rate decreased over the past few decades because of strategies such as early detection through regular screening. This involves routine mammography beyond a certain age (e.g. after 50 years old in average in France).

## Mammography: a quick explanation

Mammography is a specialized medical imaging procedure using a low dose of X-rays to allow radiologists examining the human breast. 

The breast is placed onto a flat support plate, and a parallel plate (compression paddle) presses down the tissue. Both plates are made of materials that X-rays cross easily. Compression of the breast is important, because it reduces the thickness of tissue the X-rays must cross, so fewer photons are absorbed (lower dose) or scattered with a probability that depends on the tissue. It also spreads overlapping structures apart, and it keeps the breast still during the exposure.

The produced X-rays are coming out from a focal spot (approx 0.1-0.3mm length), which can be considered as a point source, since it's size is tiny compared to distances involved. During their travel, photons can be absorbed by the breast tissues, and a fraction of these end up being absorbed by the detector.

The resulting image, called *mammogram*, is a central projection with the focal spot as center, on a plane which is the screen. Each pixel $z$ receive an amount of energy carried by incoming photons, therefore their value depend on the arrival rate $\lambda(z)$ of each photons.

# Noise in medical imaging
Noise is defined by random fluctuations around the expected value that alter the true measurement (e.g. pixels in the case of images).

Because of the nature of the material and methods used for the different imaging methods, sources of noise, and by the same way, their nature, can vary.

In the case of mammography, which relies on X-ray photons, it is in particular affected by Poisson noise.
Indeed, because the photons arrives at random times, the uncertainty on the received signal can be modeled as a Poisson distribution:

Consider a random variable $Z \sim \mathcal P(\lambda)$ describing the number of photons on a single pixels, with $\lambda$ the expected arrival photon frequency on that pixel. We have :
$$
P(z|\lambda) = \frac{\lambda^z e^{-\lambda}}{z!}
$$


The noisy image $Y$ received by the detector can be decomposed the following way:
$$
Y = \mathbb E[Y] + V
$$
where:
- $\mathbb E[Y]$: Expected image (noise free)
- $V$: Additional noise

Since $\mathbb E[Y_i] = \mathrm{Var}(Y_i) = \lambda_i$, the signal-to-noise ratio of a pixel is
$$
\mathrm{SNR} = \frac{\text{signal}}{\text{noise}} = \frac{\lambda_i}{\sqrt{\lambda_i}} = \sqrt{\lambda_i}
$$

This is because, for a given pixel photon count $Z$:
$$
\mathbb E[Z] = \lambda, \qquad \mathrm{Var}(Z) = \lambda, \qquad \sigma_Z = \sqrt{\lambda}
$$

Notice that the higher $\lambda$ is, the less noisy the image is perceived, but the higher the dose is. The purpose of this work is to keep an exploitable image, with the lowest dose of X-rays possible.

> In fact, a higher $\lambda$ gives you in absolute terms a noisier image (since $\mathrm{Var}(Z) = \lambda$), but if you consider the relative noise to the signal, it decreases (since $\sigma_Z / \mathbb E[Z] = 1/\sqrt{\lambda}$).




## Noise simulation

### Countring photons
MIAS images are film scans where dense tissue appears bright, because few photons reached the film. Then, since a pixel value is an intensity and not a photon count, we first invert them ($z = 1 - p/255$), so that a higher value means more photons.

Following the article, we will assume a linear relation between a pixel value $z$ and the photon count:
$$
z = k \cdot Z, \qquad Z \sim \mathcal P(\lambda)
$$
where $k$ is an unknown conversion factor.

Then, because $\mathbb E[z] = k\lambda$ and $\mathrm{Var}(z) = k^2\lambda$, $k$ can be estimated from the data:
$$
k = \frac{\mathrm{Var}(z)}{\mathbb E[z]}
$$
and the expected photon count of each pixel is recovered as $\lambda = z / k$.

In our case, $k$ is computed on each $128 \times 128$ training patch. Patches are drawn in breast tissue (mean value below 0.8 after inversion), so that the wide empty background areas are skipped.

### Reducing the dose
Because the number of photons is proportional to the dose. Reducing the dose by a factor $\alpha$ therefore scales the expected photon count, and we draw a new noisy image from:
$$
Z_\alpha \sim \mathcal P(\alpha\lambda), \qquad P(Z_\alpha = z) = \frac{(\alpha\lambda)^z e^{-\alpha\lambda}}{z!}
$$
The SNR of each pixel becomes $\sqrt{\alpha\lambda}$: a dose reduced by $\alpha$ divides the SNR by $\sqrt{\alpha}$ when $\alpha < 1$. The article simulates an 80% dose reduction. As in the article, a new noise realization is drawn every time a patch is loaded, so the network never sees the same noise twice.

### Anscome Transform
Poisson noise depends on the signal ($\mathrm{Var} = \lambda$), whereas our network (like BM3D) is designed to remove an additive noise of constant variance. We therefore apply the Anscombe transform:
$$
A(z) = 2\sqrt{z + \tfrac{3}{8}}
$$
It turns a Poisson variable into an approximately Gaussian one with unit variance (the approximation holds as long as the photon count is not too small). It is applied both to the noisy image $Z_\alpha$ (network input) and to the clean photon map $\alpha\lambda$ (target). The network thus learns to remove an approximately $\mathcal N(0, 1)$ noise, and the evaluation (PSNR, BM3D with $\sigma = 1$) is done in this same domain.

**Limitations of this simulation on MIAS.**
- $k = \mathrm{Var}(z)/\mathbb E[z]$ only holds if all the variance comes from Poisson noise. On a mammogram patch, the variance also contains anatomical structures (tissue texture, edges) and scanning noise. $k$ is therefore overestimated and $\lambda$ underestimated, so the "clean" patches already correspond to a low photon count. **This is most likely why we had to use $\alpha = 3.6 > 1$. In our setting**, $\alpha$ is not a real dose-reduction factor but a parameter that controls the noise level.
- The linear model $z = k \cdot Z$ suits the digital detector used in the article (125 full-field digital mammograms from Siemens). It is only an approximation for scanned films, whose response to exposure is non-linear.
- Electronic noise of the detector is not modeled, neither in the article nor here.

## The approach to solve the problem

![ResNet](figures/ResNet.png)


## Results
In order to evaluate the model performances, we computed the standard metrics for denoising tasks: the PSNR (Peak Signal-to-Noise Ratio) to evaluate the quality of the output image. It is defined by:

$$
\text{PSNR} = 10 \cdot \log_{10}\left(\frac{\text{MAX}^2}{\text{MSE}}\right)
$$
where:
- $MAX$: largest possible pixel value
- $MSE$ is the mean squared error between the generated image and the original

We obtained an average PSNR of 50.42 on the test set (while noisy image had an average of 39.19 dB), the model improves the PSNR by 11.23 dB. By the same way, we obtained a SSIM of 0.990 (0.891 for noisy images), which shows a gain of 0.099.

PSNR is computed on the Anscombe domain.

We can deduce our model is indeed able to denoise a given image.

### Limitations
the PSNR and SSIM are averages over all pixels. Which means that a denoising method that smooth the image and erase meaningful details won't be penalized as we would like to. 

To show visually show the limitation of these metrics, here is an example of denoising on microcalcifications

![Denoising limitations](figures/Denoising-microcalcifications.png)

Our model is actually oversmoothing the noisy input: microcalcifications and other structures are lost. This limitation is critical, since smoothing these crucial informations for diagnosis can lead to serious consequences for the patient. Therefore, this is a major challenge we are facing in image quality improvement in medical imaging.

### Comparison with another method: BM3D
BM3D (Block Matching and 3D Filtering) is a deterministic, non-learned algorithm commonly used for noise reduction. It is a collaborative filtering process.

| Method | PSNR (dB) | Gain (dB) | SSIM |
|---|---|---|---|
| Noisy | 39.19 | – | 0.891 |
| BM3D | **51.40** | **+12.21** | **0.991** |
| ResNet | 50.42 | +11.23 | 0.990 |

![BM3D vs ResNet](figures/bm3d-resnet.png)

BM3D outperforms the ResNet by about 1dB. Unlike the reference article where the ResNet is the most efficient algorithm, our network doesn't for several reasons: 
- The network has been trained on less data: the exact model used is trained for 95 epoch, with one random patch per image per epoch, which appxoximatively represents $\text{nb of batches per epoch} \times \text{nb of epoch} = 15 \times 95 = 1425$ steps in the optimizer. In comparison, the article has trained its model on around 40 000 images per epoch, with 125 epochs.
- The nature of our dataset makes the simulated noise less accurate, and contain wide plain uninteresting areas. This led us, in addition of memory purposes, to consider random patches over areas with a minimum amount of photons to draft the more interesting areas (human breast tissues. This simulated noise varies between each patches, because we estimated $k$ on each patches. All of these reasons made the training more difficult.

Training the model for more epochs and improbing the data quality would then be the most efficient measures to take, even tought it is still possible to optimize more the training.



## Sources
- Deep learning-based denoising of mammographic images using physics-driven data augmentation: https://arxiv.org/html/1912.05240v1#bib.bib4 
- Denoising Autoencoders | Deep Learning Animated (Youtube): https://www.youtube.com/watch?v=0V96wE7lY4w
- WHO Fact Sheet: Breast Cancer: https://www.knowledge-action-portal.com/en/content/who-fact-sheet-breast-cancer
- Loss Functions for Image Restoration with Neural
Networks: https://arxiv.org/pdf/1511.08861