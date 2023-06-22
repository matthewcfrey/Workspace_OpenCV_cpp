This folder contains functions related to feature detection of a human face.  
Functions are included to take an original image and subsample to create  
a low resolution image. The purpose of this was to experiment with image  
super resolution. Ideas included the possibility of extracting features  
from the low resolution image to blend/create details extracted from  
high resolution examples.  
Included topics are:  
- Contours
- Thresholding
- Subsampling
- Heatmaps
- Haarcascade detection

Original image:  
![person](ExampleImages/person1.jpg)  
Background removed, face identified, subsampled to create low quality:  
![background](ExampleImages/blueface.jpg)  
Eyes detected:   
![eyes](ExampleImages/both.jpg)  
Contours, all and outer:    
![all contours](ExampleImages/contours.jpg)   
![outer](ExampleImages/outerc.jpg)  
heatmap used to identify facial regions:  
![heat](ExampleImages/fheat.jpg)  
get blue color channel from heatmap, threshold the blue parts  
to identify feature locations  
![thresh](ExampleImages/bluethresh.jpg)
