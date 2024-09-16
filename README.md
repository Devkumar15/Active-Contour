# Active-Contour
Agricultural Field Boundary Detection Using Active Contours (Snakes) and Satellite Imagery  This project implements active contour models (snakes) for the automatic detection of agricultural field boundaries from satellite images. The workflow includes:  Image preprocessing with Gaussian filtering.

This project demonstrates the detection and extraction of agricultural field boundaries from satellite images using active contour models (snakes) and various image processing techniques. The project also includes geospatial processing to convert detected boundaries into georeferenced coordinates and export them in KML format for visualization in mapping tools such as Google Earth.

## Introduction
The goal of this project is to accurately extract agricultural field boundaries from satellite imagery. The core of the method is based on Active Contour Models (Snakes), a powerful technique for capturing the smooth, continuous boundaries of objects. The process involves a combination of image preprocessing, edge detection, and contour optimization using a snake-fitting algorithm. The extracted boundaries are then converted into geographic coordinates and exported as a KML file for use in geospatial mapping software.

## Features
  Active Contour Model (Snake) Implementation: Smooth boundary extraction using energy minimization techniques.
  Edge Detection: Ridge enhancement for highlighting agricultural field boundaries.
  Geospatial Integration: Conversion of pixel coordinates into geographic coordinates using GDAL.
  KML Export: Automatic generation of KML files for use with Google Earth and other geospatial tools.
  Visualization: Real-time visualization of the contour fitting process.

## Prerequisites

Before running this project, you will need to install the following dependencies:

Python 3.x
opencv-python (cv2)
rasterio
skimage (scikit-image)
matplotlib
numpy
Pillow
scipy
gdal
osgeo

### Use the following command to install the required libraries:

pip install opencv-python rasterio scikit-image matplotlib numpy Pillow scipy gdal

### Installation

#### Clone the repository:
git clone https://github.com/Devkumar15/Active-Contour.git

cd agricultural-field-boundary-detection

#### Install dependencies:
pip install -r requirements.txt

Download or use a sample satellite image in TIFF format. Place it in the project directory.

### Usage

To run the boundary detection process on a satellite image:
Place the satellite image (in .tif format) in the appropriate folder.

#### Modify the image file path in the code:
with rio.open(r"path_to_your_image.tif") as img:

#### Run the Python script:
python boundary_detection.py

Enter a filename when prompted, to save the KML file.

## Workflow

### 1. Importing Satellite Image
The code begins by loading a satellite image in .tif format using rasterio, converting it to a NumPy array for further processing.

### 2. Image Preprocessing
Gaussian Filter: A Gaussian filter is applied to the image to reduce noise and smooth the image.

### 3. Edge Detection and Ridge Enhancement
Ridge Detection: The image is enhanced to emphasize the ridges (which likely represent field boundaries) using the Hessian matrix method from skimage.
Skeletonization: The detected ridges are converted into a skeleton, reducing the boundaries to thin lines.

### 4. Active Contour (Snake) Fitting
The active contour model is applied to fit a snake (a continuous, smooth curve) around the detected field boundaries. This process optimizes the snake by minimizing the energy, balancing edge energy, spacing, and curvature.
Snake Energy: The algorithm uses internal and external energy minimization to optimize the snake's fit to the field boundaries.

### 5. Conversion to Geographic Coordinates
Pixel coordinates of the detected boundary points are converted into geospatial coordinates using the GDAL library. This transformation allows the coordinates to be mapped in real-world geographic systems.
### 6. KML File Generation
The code generates a KML file from the geographic coordinates, allowing the boundaries to be visualized in tools like Google Earth.

##### Enter filename: boundaries
A KML file named boundaries.kml will be generated in the specified directory, which can be imported into Google Earth or other GIS platforms.

### Results
The KML file generated from this process can be opened in Google Earth to visualize the agricultural fields with high precision. The boundaries extracted from the satellite image are represented as polygons overlaid on the map.

Example Input:

![og](https://github.com/user-attachments/assets/2376dd56-f993-4b99-8f2b-0830000e5aa4)

Example Output (Google Earth):

![kml](https://github.com/user-attachments/assets/666f96c5-c4a1-4bd2-907f-a99620e901c5)


### Future Enhancements

Potential improvements to this project include:
#### Multi-field Detection: Enhance the algorithm to detect multiple fields in a single image.
#### Improved Edge Detection: Experiment with advanced edge-detection algorithms for better accuracy.
#### Support for Additional Formats: Expand the input support to include other formats such as GeoTIFF, JPEG2000, etc.

### Contributing
Contributions are welcome! Please submit a pull request or open an issue to suggest improvements or report bugs.
