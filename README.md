# change_detection_memorability
*Pre-existing semantic associations contribute to memorability of visual changes. In preparation, 2022*

Code underlying the deep image embedding results for the above in-preparation manuscript. These python scripts were developed in partial fulfillment of "EN.580.437: Biomedical Data Design" and in collaboration with the NINDS Functional Neurosurgery Lab. 

## Key Result
The results show that the "added" order of presentation tends to be less memorable than the "removed" order when the image pair is more closely semantically related. Semantic relatedness is quantified using L2 distance, cosine similarity, and dot product on the penultimate EfficientNetB0 activation.

![image](https://user-images.githubusercontent.com/98730743/207410128-c1e6a251-0520-49ef-b345-0f3eec9a59d2.png)

## Contents
- The custom dataset of ~1,024 image pairs used in the change detection test is not publicly available
- network_modeling.ipynb: Python notebook with all experiment
  - Module scripts: image_embedding.py, memorability_prediction.py

