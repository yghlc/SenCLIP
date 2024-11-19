# SenCLIP: Enhancing zero-shot land-use mapping for Sentinel-2 with ground-level prompting
<div align="center">
<img src="model_arch.jpg" width="1000" height="500">

[Pallavi Jain*](https://scholar.google.com/citations?user=MMYyjyIAAAAJ&hl=en),
[Dino Ienco](https://scholar.google.com/citations?hl=en&user=C8zfH3kAAAAJ),
[Roberto Interdonato](https://scholar.google.com/citations?user=GWACYGoAAAAJ&hl=en),
[Tristan Berchoux](https://scholar.google.com/citations?hl=en&user=shdhPjcAAAAJ) and
[Diego Marcos](https://scholar.google.com/citations?user=IUqydU0AAAAJ&hl=en)
</div>
[Accepted at WACV'25.]
[Model Checkpoints]
[Sentinel-2 Dataset]


This repository implements **SenCLIP**, a vision-language framework that adapts **CLIP** for zero-shot land-use/land-cover (LULC) mapping with Sentinel-2 imagery. Pre-trained models like CLIP excel at zero-shot tasks but struggle with satellite imagery due to limited representation in their training data.  

SenCLIP addresses this gap by aligning Sentinel-2 satellite images with geotagged [LUCAS 2018](https://ec.europa.eu/eurostat/web/lucas/database/2018) ground-level photos, enabling the model to understand detailed ground-view descriptions. Evaluated on EuroSAT and BigEarthNet, SenCLIP significantly improves LULC classification accuracy using both aerial and ground-level prompts, advancing the use of vision-language models for medium-resolution satellite imagery.
