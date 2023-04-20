# PDAtt-Unet: Pyramid Dual-Decoder Attention Unet For Covid-19 Infection Segmentation from CT-scans.

In summary, the main contributions of this paper are as follows:

- Inspired by the Att-Unet architecture, we propose three different architectures for segmenting Covid-19 infections from CT-scans. The first variant, Pyramid Att-Unet (PAtt-Unet), uses image pyramids to preserve the spatial awareness in all of the encoder layers. Unlike most attention-based segmentation architectures, our proposed PAtt-Unet uses the attention gates not only in the decoder but also in the encoder.

- Based on PAtt-Unet and DAtt-Unet, we propose a Pyramid Dual-Decoder Att-Unet (PDAtt-Unet) architecture using the pyramid and attention gates to preserve the global spatial awareness in all of the encoder layers. In the decoding phase, PDAtt-Unet has two independent decoders that use the Attention Gates to segment infection and lung simultaneously.

- To address the shortcomings of the binary cross entropy loss function in distinguishing the infection boundaries and the small infection regions, we propose the ${BCE}_{Edge}$ loss that focuses on the edges of the infection regions.

- To evaluate the performance of our proposed architectures, we use four public datasets with two evaluation scenarios (intra and cross datasets),  all slices from CT scans are used for the training and testing phases. 

- To compare the performance of our approach with other CNN-based segmentation architectures, we use three baseline architectures (Unet, Att-Unet and Unet++) and three state-of-the-art architectures for Covid-19 segmentation (InfNet, SCOATNet, and nCoVSegNet). The experimental results show the superiority of our proposed architecture compared to the basic segmentation architectures as well as to the three state-of-the-art architectures in both intra-database and inter-database evaluation scenarios.


![PDEAttUnet (1)](https://user-images.githubusercontent.com/18519110/228053614-95a1574a-5c8a-45f2-a0d0-f30590474a2f.png)

<p align="center">
  Figure 1: Our proposed PDEAtt-Unet architecture details.
</p> 

## Implementation:
#### PDAtt-Unet architecture and Hybrid loss function:
``` Architectures.py ``` contains our implementation of the comparison CNN baseline architectures  (Unet, Att-Unet and Unet++) and the proposed PDAtt-Unet. architecture.

``` Hybrid_loss.py ``` contains the proposed Edge loss function.

#### Training and Testing Implementation:
``` detailed train and test ``` contains the training and testing implementation.

- First: the dataset should be prepared using ``` prepare_dataset.py ```, this saves the input slices, lung mask, and infection mask as ``` .pt ``` files
The datasets could be donwloaded from: http://medicalsegmentation.com/covid19/

- Second:  ``` train_test_PDAttUnet.py ``` can be used to train and test the proposed PDAtt-Unet architecture with the proposed Hybrid loss function (with Edge loss).


## Citation: If you found this Repository useful, please cite:

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
## Ablation architectures: PAttUnet and DAtt-Unet 


![PAttUnet (1)](https://user-images.githubusercontent.com/18519110/164985902-fbf77196-e435-40ec-aa89-bdeb1cdfc093.png)
<p align="center">
  Figure 2: Our proposed PAtt-Unet architecture details.
</p>

![CDAttUnet (1)](https://user-images.githubusercontent.com/18519110/164985900-d1b48555-8a6d-4bb0-86f8-d8ddf7b415df.png)
<p align="center">
  Figure 3: Our proposed DAtt-Unet architecture details.
</p>  

