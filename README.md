# Face-Frontalisation
---  
This is a GAN architecture trained to generate images with frontal views of a face given a face turned sideways. It utilizes an autoencoder model. The idea is to extract structural features of frontal face from autoencoder and fuse it with facial features in upsampling part of GAN.
![architecture][/output/architecture.jpg "Model Architecture"]  

###### Identity Loss  
LightCNN is used to extract the facial features and calculate the identity between the generated image and the ground truth.  
The trained LightCNN can be downloaded from [here](https://github.com/AlfredXiangWu/LightCNN).

#### Sample Output  
![sample output](/output/test.jpg "Sample Output")  

Trained models are (here)[https://drive.google.com/drive/folders/1Op1ShbpjPZg2ncXlblVPWKuETv7lOzrE?usp=sharing]
