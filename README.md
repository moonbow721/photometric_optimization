# Photometric FLAME Fitting

This repository provides an analysis-by-synthesis framework to fit a textured [FLAME](http://flame.is.tue.mpg.de/) model to an image. FLAME is a lightweight generic 3D head model learned from over 33,000 head scans, but it does not come with an appearance space (see the [scientific publication](https://ps.is.tuebingen.mpg.de/uploads_file/attachment/attachment/400/paper.pdf) for details). 

<p align="center"> 
<img src="images/tex_0_-3.0.png" width="150"/>
<img src="images/tex_1_-3.0.png" width="150"/>
<img src="images/tex_2_-3.0.png" width="150"/>
<img src="images/tex_3_-3.0.png" width="150"/>  
<img src="images/tex_4_-3.0.png" width="150"/>  
</p>
<p align="center"> 
<img src="images/tex_0_3.0.png" width="150"/>
<img src="images/tex_1_3.0.png" width="150"/>
<img src="images/tex_2_3.0.png" width="150"/>
<img src="images/tex_3_3.0.png" width="150"/>  
<img src="images/tex_4_3.0.png" width="150"/>  
</p>
<p style="text-align: justify;"><em><font color="grey">Variations of the texture space for the first five principal components. Each column shows the variation for &plusmn;2 standard deviations along one axis.</font></em></p>
 
This repository 
1) describes how to build a texture space for FLAME from in-the-wild images, and provides
2) code to fit a textured FLAME model to in-the-wild images, optimizing for FLAME's parameters, appearance, and lighting, and
3) code to optimize for the FLAME texture to match an in-the-wild image. 

**The FLAME model and the texture space can be downloaded from the [FLAME project website](https://flame.is.tue.mpg.de). You need to sign up and agree to the license for access.**

**The demos will be released soon.**

## Using the DPoser Prior



### 1. Prerequisites

Before you begin, please complete the following setup steps:

* **Download Models:** First, download the required model files for this repository from this [**Google Drive link**](https://drive.google.com/file/d/1i65D5Sa46aiiaoj1txYaypn-l0IHYd6s/view?usp=drive_link).

* **Configure DPoser-X:** This project depends on our **[DPoser-X repository](https://github.com/moonbow721/DPoser-X)**. Please ensure you have cloned it and correctly configured its Python environment and model checkpoints. The `prior.py` script dynamically imports DPoser modules by appending its directory to the system path, so a complete DPoser-X setup is essential for it to work.

---

### 2. Running the Demo

To run the main demo script with the DPoser prior and generate a video of the fitting process, execute the following command:

```bash
python photometric_fitting.py --prior DPoser --device cuda:0 --save-video True
```
This will run the optimization on the sample images provided and save the results, including a vis_video.avi file, in the output directory.

### 3. Processing Custom Images
You can easily adapt the script to process your own images. The only requirement is to provide facial landmark files that correspond to your images.

You can generate the required JSON landmark files using the excellent pre-trained models available in [MMPose](https://mmpose.readthedocs.io/en/0.x/topics/face.html). Simply process your images with an MMPose face model and use the resulting JSON output as input for our script.

## Build FLAME texture space from in-the-wild images

The goal is to build a texture space from in-the-wild images in order to cover large range of ethnicities, age groups, etc. We therefore randomly select 1500 images from the [FFHQ dataset](https://github.com/NVlabs/ffhq-dataset) in order to build a texture space. This is done in following steps

***1. Initialization***

Building a texture space from in-the-wild images is a chicken-and-egg problem. Given a texture space, it can be used in an analysis-by-synthesis fashion to fit the 3D model to images, where these fits then can be used to build a texture space. To get an initial texture space, we fit FLAME to the [Basel Face Model (BFM)](https://faces.dmi.unibas.ch/bfm/index.php?nav=1-0&id=basel_face_model) template, and project the BFM vertex colors onto the FLAME mesh, to get an initial texture basis.

***2. Model fitting***

We then fit FLAME to the FFHQ images, optimizing for the FLAME shape, pose, and expression parameters, the parameters of the initial texture space, the parameters for Spherical Harmonics (SH) lighting (we optimize for 9 SH coefficient only, shared across all three color channels), and a texture offset to capture texture details deviating from the initial texture space. The fitting minimizes a landmark loss, a photometric loss, and diverse regularizers for shape, pose, expression, appearance, and the texture offset. 

The landmark loss minimizes the difference between the landmarks projected from the face model's surface, and predicted 2D landmarks (predicted using the [FAN landmark predictor](https://github.com/1adrianb/face-alignment)). The photometric loss is optimized for the skin region only (provided by the [face segmentation network](https://github.com/YuvalNirkin/face_segmentation)) to gain robustness to partial occlusions. See the provided code for details how to fit a textured FLAME model to an image. 

***3. Texture completion***

After fitting, the computed texture offsets capture for each image the facial appearance of the non-occluded skin region. To complete the texture maps, we train an inpainting network adapted from [GMCNN](https://github.com/shepnerd/inpainting_gmcnn) (across all texture maps) supervisely by adding random strokes (i.e. strokes of random size and location) in the visible face region(visibility obtained from the fitted reconstruction) and learning to inpaint these strokes. Once trained, we inpaint all missing regions with the resulting inpainting network.

***4. Texture space computation***

After completing these 1500 texture maps, we use principal component analysis (PCA) to compute a texture space. 

## Demos
The single image photometric fitting demo is implemented and tested in a conda environment with PyTorch 1.5 and PyTorch3D 0.2 in Python 3.8.3. For better CUDA supports, we recommend you to install PyTorch3D 0.2 via conda, 

```
conda create -n pytorch3d python=3.8
conda activate pytorch3d
conda install -c pytorch pytorch=1.5.0 torchvision cudatoolkit=10.2
conda install -c conda-forge -c fvcore fvcore
conda install pytorch3d -c pytorch3d
```

ATTENTION: The pip and conda packages of PyTorch3D have different dependencies, please follow their installation guide.

Run this demo with specified FFHQ image name and computing device,
```
python photometric_fitting.py 00000 cuda
```

Another simple demo to sample the texture space can be found [here](https://github.com/TimoBolkart/TF_FLAME).


## License
The code is available for non-commercial scientific research purposes. The texture model is available under [Creative Commons BY-NC-SA 4.0 license](https://creativecommons.org/licenses/by-nc-sa/4.0/). For details see the [Texture license](https://flame.is.tue.mpg.de/texturelicense.html).

## Notes
We use the FLAME.py from [FLAME_PyTorch](https://github.com/soubhiksanyal/FLAME_PyTorch) and the renderer.py from [DECA](https://github.com/YadiraF/DECA).

## Citation

When using this code or the texture model in a scientific publication, please cite **this GitHub repository** and the **FFHQ dataset**. When using the FLAME geometry model, please cite the model (you find the up-to-date bibtex [here](https://flame.is.tue.mpg.de/)).

## Contact
For questions regarding the provided fitting code please contact haiwen.feng@tuebingen.mpg.de, for FLAME related questions please contact flame@tuebingen.mpg.de.
