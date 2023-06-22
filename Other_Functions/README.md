The project includes
- discrete fourier transforms and work in frequency domain
- low pass and high pass filters
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
