# Earth on Canvas dataset
### A Zero-Shot Sketch-based Inter-Modal Object Retrieval Scheme for Remote Sensing Images 

Earth on Canvas dataset Page: (https://ushasi.github.io/Earth-on-Canvas-dataset/)


1. Airplane
2. Baseball Diamond
3. Buildings
4. Freeway
5. Golf Course
6. Harbor
7. Intersection
8. Mobile home park
9. Overpass
10. Parking lot
11. River
12. Runway
13. Storage tank
14. Tennis court

To implement the code:
<ol>

<li> If needed change the path of snapshots and summary folders by changing the ‘path’ variable in unified_SI.py

<li> For pretraining the X and Y modalities, get the code from the pre-training git repository and load this .mat file in the <b>UxUyLoader.py</b> file. We have also added a sample pretrained features in data folder. </li>

<!--- <li> While in the master folder, run the <b>Unified_XY_triads.py</b> file (for terminal based, type ‘python Unified_XY_triads.py’ in terminal) </li> --->

<!--- <li> unified_features.mat is the features of the entire dataset, obtained by training just on the seen classes.  A k-NN distance of the unseen samples can be made from these features to obtain the precision, recall, and mAP values of the proposed framework.  </li>
</ol> --->





### Paper

*    The paper is also available on ArXiv: [A Zero-Shot Sketch-based Inter-Modal Object Retrieval Scheme for Remote Sensing Images](https://arxiv.org/pdf/2008.05225.pdf)

*   Feel free to cite the author, if the work is any help to you:

```
@InProceedings{Chaudhuri_2020_EoC,
author = {Chaudhuri, Ushasi and Banerjee, Biplab and Bhattacharya, Avik and Datcu, Mihai},
title = {A Zero-Shot Sketch-based Inter-Modal Object Retrieval Scheme for Remote Sensing Images},
booktitle = {http://arxiv.org/abs/2008.05225},
month = {Aug},
year = {2020}
} 
