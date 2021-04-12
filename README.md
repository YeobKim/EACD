# EACD
Deblurring Network Using Edge Module, ASPP Channel Attention and Dual Network (NTIRE 2021 Challenge)

This is a PyTorch implementation of the [NTIRE 2021 Image Deblurring Challenge - Track2. JPEG Artifacts](https://competitions.codalab.org/competitions/28074)

## Abstract
In this work, we propose an edge module, ASPP Residential Channel Attachment Blocks, and a network using Dual Network for image deblurring. The proposed network extracts and utilizes edge information for learning through modules that extract edges inside the entire network. By extracting edges and using them for learning, the network is constructed to preserve details when deblurring. In particular, proposed network is very fast at 0.012 seconds to process a 1k-sized validation set. It is very fast in processing and is useful for real-time deblurring systems, such as autonomous vehicles. We participated in NTIRE 2021 Image Deblurring Challenge - Track2. JPEG Artifacts, showing results of PSNR 27.00, SSIM 0.7670 and 0.012 runtime per image on the validation set. 

## Proposed Algorithm
- Network Architecture
![model_architecture](https://user-images.githubusercontent.com/59470033/111621750-b4030480-882b-11eb-857c-715781ca7d84.png)

- Edge Module & Feature Block
![Edge Feature](https://user-images.githubusercontent.com/59470033/111582406-1e04b500-87fe-11eb-9ddf-62b308c8fd21.png)

- ARCAB(ASPP Residual Channel Attention Block)
![ARCAB](https://user-images.githubusercontent.com/59470033/111582004-8c954300-87fd-11eb-97ce-f97d836ce52a.png)

- RDB(Residual Dense Block) & RG(Residual Group)
![RDB_RG](https://user-images.githubusercontent.com/59470033/111582668-80f64c00-87fe-11eb-9480-4de0140e567e.png)

## Run
'''c
python test.py
'''

## Dataset
We used the Datasets provided by [NTIRE 2021 challenge](https://data.vision.ee.ethz.ch/cvl/ntire21/) for training the proposed network.

## Experimental Rsults
### Validation
- PSNR - 27.00
- SSIM - 0.7670
- Run Time per image - 0.012
### Test
- PSNR - 27.41
- SSIM - 0.7887
- Run Time per image - 0.012

## Contact
If you have any questions, please contact athurk94111@gmail.com.
