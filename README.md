# Image-Resizer


This project implements the seam carving algorithm in C++ using OpenCV. Seam carving is a content-aware image resizing technique that removes low-energy vertical or horizontal seams to reduce the image size while preserving important content.

## Features

- Calculates energy map using gradient magnitude
- Finds and removes vertical and horizontal seams
- Resizes images while preserving significant content
- Works with most image formats supported by OpenCV

### Install Packages (Ubuntu)

```bash
sudo apt-get update
sudo apt install pkg-config
```
```bash
sudo apt-get install libopencv-dev
```

###To run the code

In your run.sh

```bash
g++ -o seam_carving seam_carving.cpp `pkg-config --cflags --libs opencv4`
```
After completing the steps above, compile using:

```bash
./run.sh
```
```bash
./2024202010_A1_Q4 image_path
```
