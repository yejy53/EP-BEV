# EP-BEV
About The official implementation of the paper "Cross-view image geo-localization with Panorama-BEV Co-Retrieval Network“. (ECCV 2024)

Cross-view geolocalization identifies the geographic location of street view images by matching them with a georeferenced satellite database. Significant challenges arise due to the drastic appearance and geometry differences between views. In this paper, we propose a new approach for cross-view image geo-localization, i.e.,  the Panorama-BEV Co-Retrieval Network. Specifically, by utilizing the ground plane assumption and geometric relations, we convert street view panorama images into the BEV view, reducing the gap between street panoramas and satellite imagery. In the existing retrieval of street view panorama images and satellite images, we introduce BEV and satellite image retrieval branches for collaborative retrieval. By retaining the original street view retrieval branch, we overcome the limited perception range issue of BEV representation. Our network enables comprehensive perception of both the global layout and local details around the street view capture locations. Additionally, we introduce CVGlobal, a global cross-view dataset that is closer to real-world scenarios. This dataset adopts a more realistic setup, with street view directions not aligned with satellite images. CVGlobal also includes cross-regional, cross-temporal, and street view to map retrieval tests, enabling a comprehensive evaluation of algorithm performance.

## News
- 2024/08, source code of BEV transformation is released（CVACT/CVUSA）.

# Data preparation

The CVGlobal dataset can be downloaded from "https://opendatalab.com/CVeRS/CVGlobal" or "https://huggingface.co/datasets/Yejy53/CVGlobal".



![ECCV2](https://github.com/user-attachments/assets/02252a74-a116-4829-80af-96f2426a326a)

## References
We appreciate the previous open-source works.
* [Boosting3DoF]([https://github.com/YujiaoShi/Boosting3DoFAccuracy])
