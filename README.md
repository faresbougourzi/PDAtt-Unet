# PDAtt-Unet: Pyramid Dual-Decoder Attention Unet For Covid-19 Infection Segmentation from CT-scans.

In summary, the main contributions of this paper are as follows:

- We propose ParamSmoothL1 regression loss function. In addition, we introduce a
dynamic law that changes the parameter of the robust loss function during train-
ing. To this end, we use the cosine law with the following robust loss functions:
ParamSmoothL1, Huber and Tukey. This can solve the issue of complexity in
searching the best loss function parameter.

- We propose two branches network (REX-INCEP) for facial beauty estimation based
on ResneXt-50 and Inception-v3 architectures. The main advantage of our REX-
INCEP architecture is its ability to learn high-level FBP features using ResneXt and
Inception blocks simultaneously, which proved its efficiency compared to seven CNN
architectures. Moreover, our REX-INCEP architecture provides the right tradeoff
between the performance and the number of parameters for facial beauty prediction.

- We propose ensemble regression for facial beauty estimation by fusing the predicted
scores of one branch networks (ResneXt-50 and Inception-v3) and two branches
network (REX-INCEP) which are trained using four loss functions (MSE, dynamic
ParamSmoothL1, dynamic Huber and dynamic Tukey). Although the individual
regression models are trained using the same fixed hyper-parameters, the resulting
ensemble regression yields the best accurate estimates compared to the individual
models and to the state-of-the-art solutions.

![CDAttUnet (1)]![1-s2 0-S1361841523000580-gr3](https://user-images.githubusercontent.com/18519110/228052439-693624f2-8abe-4fa4-b5fb-903c9d37b7f2.jpg)
<p align="center">
  Figure 1: Our proposed DAtt-Unet architecture details.
</p>  

## Citation: If you found this  this Repository useful, please cite:

```bash
@article{bougourzi2023pdatt,
  title={PDAtt-Unet: Pyramid Dual-Decoder Attention Unet for Covid-19 infection segmentation from CT-scans},
  author={Bougourzi, Fares and Distante, Cosimo and Dornaika, Fadi and Taleb-Ahmed, Abdelmalik},
  journal={Medical Image Analysis},
  pages={102797},
  year={2023},
  publisher={Elsevier}
}
```


![PAttUnet (1)](https://user-images.githubusercontent.com/18519110/164985902-fbf77196-e435-40ec-aa89-bdeb1cdfc093.png)
<p align="center">
  Figure 2: Our proposed PAtt-Unet architecture details.
</p>

![CDAttUnet (1)](https://user-images.githubusercontent.com/18519110/164985900-d1b48555-8a6d-4bb0-86f8-d8ddf7b415df.png)
<p align="center">
  Figure 3: Our proposed DAtt-Unet architecture details.
</p>  

