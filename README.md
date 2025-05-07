# Dlib-ImageNet-Datasets  
**Preprocessed Stable ImageNet-1K datasets for efficient computer vision prototyping with Dlib.**  

## 📌 Overview  
This repository provides **ready-to-use ImageNet-1K datasets** preprocessed in multiple resolutions (32×32 to 256×256) for the [Dlib](http://dlib.net/) machine learning library. Designed for rapid experimentation, benchmarking, and model training, these datasets eliminate preprocessing overhead while ensuring consistency across experiments.  

🚀 **Immediately available**:  
- A ready-to-use **32×32 resolution dataset** (ideal for lightweight model prototyping) in the /dataset directory.

🛠️ **Flexible generation**:  
The included C++14 tool lets you create custom datasets in **any resolution** (e.g., 64×64, 128×128, 224×224, etc.) from raw ImageNet-1K sources. Perfect for:  
- Rapid experimentation  
- Resolution-impact benchmarking  
- Consistent model training pipelines

## Create Custom Datasets
Compile and run the included tool to process raw ImageNet-1K images:
```cpp
g++ -std=c++14 src/create_dataset.cpp -o create_dataset -ldlib -lpthread
./create_dataset path/to/imagenet_root datasets/128x128/imagenet_128.dat 128
```

## Evaluate Models
Load pre-split training/testing sets:
```cpp
std::vector<dlib::matrix<rgb_pixel>> train_images, test_images;
std::vector<unsigned long> train_labels, test_labels;
dlib::load_stable_imagenet_1k("datasets/64x64/imagenet_64.dat", train_images, train_labels, test_images, test_labels);
```
