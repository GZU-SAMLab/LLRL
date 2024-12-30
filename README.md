# LLRL
Location-guided Lesions Representation Learning via Image Generate for Assessing Plant Leaf Diseases Severity
# Abstract
Accurate assessment of plant leaf disease severity is crucial for implementing precision pesticide application, which in turn significantly enhances crop yields. Previous methods primarily rely on global perceptual learning, often leading to the misidentification of non-lesion regions as lesions within complex backgrounds, thereby compromising model accuracy. To address the challenge of background interference, we propose a novel location-guided lesion representation learning method (LLRL) based on image generation to assess the severity of plant leaf diseases. Our approach comprises three key networks: the image generation network (IG-Net), the location-guided lesion representation learning network (LGR-Net), and the hierarchical lesion fusion assessment network (HLFA-Net). IG-Net is designed to construct paired images necessary for LGR-Net by utilizing a diffusion model to generate diseased leaves from healthy ones. First, the LGR-Net facilitates the network's focus on the lesion area by contrasting paired images: healthy and diseased leaves, obtaining a pre-trained dual-branch feature encoder (DBF-Enc) that incorporates lesion-specific prior knowledge, providing focused visual features for HLFA-Net. Second, the HLFA-Net, which shares and freezes the DBF-Enc, further fuses and optimizes the features extracted by DBF-Enc, culminating in a precise classification of disease severity. In addition, we construct an image dataset containing three plant leaf diseases from apple, potato, and tomato plants, with a total of 12,098 photos, to evaluate our approach. Finally, experimental results demonstrate that our method outperforms existing classification models, with at least an improvement of 1\% in accuracy for severity assessment, underscoring the efficacy of the LLRL method in accurately identifying the severity of plant leaf diseases.

# Framework
![Framework](imgs/overview.jpg)

# LGR-Net
![LGR-Net](imgs/LGR-Net.jpg)

# HLFA-Net
![HLFA-Net](imgs/HLFA-Net.png)

# Environment
You can create a new Conda environment by running the following command:
```
    pip install -r requirements.txt
```

# Dataset
Only the test set of the plant leaf disease severity assessment dataset is publicly available, but the training set will be made public when the paper is accepted. Download the dataset to the './dataset' folder
<ul>
https://drive.google.com/drive/folders/1i2Mg7n8l7X1kB8egwESNqoOSWZFn4pu6?usp=sharing
</ul>

# PreTrained Model
The pre-trained ResNet18, LGR-Net, and HLFA-Net models are linked below, and download them to the './weight' folder. You can download the data set and the pre-training weights in the following links:
http://llrl.samlab.cn/

# Train
Train LGR-Net.
```
python ./main/mainBackBone.py \
        --file_root <The path of healthy-diseased paired images> \
        --max_steps 40000 \
        --batch_size 16 --lr 2e-4 --gpu_id 0
```
Train HLFA-Net.
```
python ./main/mainClassifier.py --num_class 6 --epochs 100 \
      --data-path ./dataset \
      --model-path <The path of the weights saved by pre-trained LGR-Net>
```

# Inference
```
python ./main/mainClassifierVal.py --num_class 6 \
      --data-path ./dataset \
      --model-path <The path of the weights of HLFA-Net>
```


# Result
![result](imgs/result.jpg)


The website with relevant details of the paper: http://llrl.samlab.cn/home.html
