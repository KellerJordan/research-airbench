"""
svd_airbench.py

Variant of clean_airbench which uses a slow SVD-based optimizer.

First, always with renormalizing every filter to have unit norm:
* If you use normal warmup, then attains 94.00(n=80) accuracy in only 8 epochs.
* If you use no warmup at all, then attains 94.03(n=96).
* If you use no warmup for filters and normal warmup for norm biases etc, then attains 94.02(n=96).
Now moving to no warmup at all by default:
* Renormalizing the entire layer to have sqrt(channels_out) norm: 94.06(n=120)
Now moving to renormalize the entire layer to have sqrt(channels_out) by default:
* Learning rate=0.25: 94.04(n=24)
* Just use normal grad, don't even do the SVD: 76.9(n=24)
* Just use normal grad, but divide by top singular value: 90.95(n=16)
* Divide by top singular value, then replace just top half of singular values by 1: 93.66(n=16)
* Sanity check: replace all singular values by 1: 94.16(n=8)
* Divide by top singular value, then sqrt all the singular values: 93.10(n=8)
* Same then fourth root: 93.76(n=8)
Now some optimizer hyperparam experiments:
* lr=0.07 momentum=0.85 nesterov=True -> 93.94(n=8)
* lr=0.10 momentum=0.85 nesterov=True -> 93.91(n=8)
* lr=0.20 momentum=0.85 nesterov=True -> 93.31(n=8)

* lr=0.12 momentum=0.70 nesterov=True -> 94.01(n=8)
* lr=0.15 momentum=0.70 nesterov=True -> 94.06(n=8)

* lr=0.12 momentum=0.60 nesterov=True -> 94.07(n=16)
* lr=0.135 momentum=0.60 nesterov=True -> 94.15(n=8)
* lr=0.15 momentum=0.60 nesterov=True -> 94.15(n=72) [best]
* lr=0.16 momentum=0.60 nesterov=True -> 94.13(n=24)
* lr=0.175 momentum=0.60 nesterov=True -> 94.04(n=8)
* lr=0.20 momentum=0.60 nesterov=True -> 94.00(n=24)

* lr=0.15 momentum=0.50 nesterov=True -> 94.09(n=16)
* lr=0.16 momentum=0.50 nesterov=True -> 94.13(n=16)
* lr=0.175 momentum=0.50 nesterov=True -> 94.12(n=16)
* lr=0.18 momentum=0.50 nesterov=True -> 94.14(n=48)
* lr=0.20 momentum=0.50 nesterov=True -> 94.13(n=64)
----
* lr=0.12 momentum=0.60 nesterov=False -> 94.11(n=8)
* lr=0.15 momentum=0.60 nesterov=False -> 93.99(n=8)

New defaults: lr=0.15 momentum=0.60 nesterov=True
(* Epochs=8 -> 94.15(n=72))
* Epochs=7 -> 93.95(n=160)
* Epochs=10 -> 94.35(n=8)
* Epochs=20 -> 94.64(n=8)
* Epochs=30 -> 94.78(n=8)
* Epochs=40 -> 94.71(n=8) [Note that the default optimizer also peaks at ~94.80, roughly at the same time?]

New defaults: that with epochs=7 (-> 93.95(n=160))
* Always add 0.1 to learning rate in scheduler (so peak is 1.1x and bottom is 0.1x) -> 93.50(n=40)
* Replace the second half of singular values with zero -> 93.57(n=8)
* Replace the second half of singular values with themselves divided by the median singular value, rather than with 1.0 -> 93.97(n=128)
* Replace the last 25% of singular values with themselves divided by the 75%ile value, rather than with 1.0 -> 93.99(n=16)
* Replace the last 75% of singular values with themselves divided by the 25%ile value, rather than with 1.0 -> 93.82(n=16)
* Replace the last 75% of singular values with themselves divided by the 25%ile value, rather than with 1.0; and then sqrt the last 75% -> 93.95(n=16)

Longer training experiments:
* Cutout=10 Epochs=30 -> 94.97(n=8) [Note that the default optimizer gets 94.76(n=5) in this setup]
* Cutout=16 Translate=4 Epochs=80 -> 95.09(n=8) [Note that the default optimier gets 94.95(n=3)]

Batch size experiments:
* Bs=500 -> 93.73(n=8)
* Bs=500 lr=0.10 -> 93.90(n=8)
* Bs=500 lr=0.12 -> 93.92(n=8)
* Bs=2000 -> 93.58(n=8)
* Bs=2000 lr=0.18 -> 93.68(n=16)
* Bs=2000 lr=0.20 -> 93.73(n=16)
* Bs=2000 lr=0.24 -> 93.73(n=32)
* Bs=2000 lr=0.30 -> 96.62(n=16)
* Bs=2000 lr=0.24 momentum=0.5 -> 93.76(n=32)
* Bs=2000 lr=0.30 momentum=0.5 -> 93.68(n=16)
* Bs=2000 lr=0.24 momentum=0.4 -> 93.64(n=16)
* Bs=2000 lr=0.30 momentum=0.4 -> 94.70(n=16)

* Bs=2000 lr=0.24 Epochs=8 -> 93.94(n=8)
* Bs=2000 lr=0.24 Epochs=8 bias_lr=5.0 -> 93.98(n=40)
* Bs=2000 lr=0.24 Epochs=8 bias_lr=6.5 -> 94.04(n=40)
* Bs=2000 lr=0.20 Epochs=8 bias_lr=6.5 -> 94.01(n=40)

* Bs=2500 lr=0.24 Epochs=8 -> 93.83(n=8)
* Bs=2500 lr=0.24 Epochs=8 bias_lr=5.0 -> 93.94(n=40) [Reducing the bias lr becomes very important at large batch size!]
* Bs=2500 lr=0.24 Epochs=8 bias_lr=6.0 -> 93.91(n=40)
* Bs=2500 lr=0.24 Epochs=8 bias_lr=6.0 momentum=0.70 -> 93.87(n=40)
* Bs=2500 lr=0.24 Epochs=8 bias_lr=5.0 wd=0.010 -> 93.96(n=40)

* Bs=5000 lr=0.24 Epochs=12 bias_lr=2.5 -> 93.94(n=48)
* Bs=5000 lr=0.30 Epochs=12 bias_lr=2.5 -> 93.98(n=64)
* Bs=5000 lr=0.30 Epochs=12 bias_lr=4.0 -> 94.04(n=88)

* Bs=10000 lr=0.30 Epochs=24 bias_lr=2.0 -> 94.12(n=32)
* Bs=10000 lr=0.30 Epochs=18 bias_lr=2.0 -> 93.67(n=32)
* Bs=10000 lr=0.24 Epochs=18 bias_lr=2.0 -> 93.50(n=32)
* Bs=10000 lr=0.40 Epochs=18 bias_lr=2.0 -> 93.42(n=32)
* Bs=10000 lr=0.30 Epochs=18 bias_lr=3.0 -> 93.52(n=24)
* Bs=10000 lr=0.30 Epochs=18 bias_lr=2.0 momentum=0.40 -> 93.42(n=24)
* Bs=10000 lr=0.30 Epochs=20 bias_lr=2.0 -> 93.84(n=24)
* Bs=10000 lr=0.30 Epochs=22 bias_lr=2.0 -> 94.05(n=24) [wow!]
* Bs=10000 lr=0.30 Epochs=22 bias_lr=2.0 bias_scaler=16.0 -> 93.88(n=24)
* Bs=12500 lr=0.30 Epochs=27 bias_lr=1.6 -> 94.00(n=24)
* Bs=12500 lr=0.30 Epochs=25 bias_lr=1.6 -> 93.80(n=24)

It is evident that going from bs=10000 to bs=12500 does not improve
the quality of each step. We still need the same number of steps to reach 94.
With either one, we can reach 94 in about 110 steps.
And with bs=5000, similarly we can reach 94 in 12 epochs (120 steps).
This is dramatically better than what can be obtained with a more first order optimizer.

New defaults: bs=2000 lr=0.24 epochs=8 bias_lr=6.5 momentum=0.6 nesterov=True
* -> 94.01(n=320)
* Momentum=0.5 lr=0.3 -> 93.93(n=16)

"""

