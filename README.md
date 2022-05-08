# Sign_Language_Recognition_with_GUI

Deep learning, and computer vision can be used too to make an impact on this cause.

This can be very helpful for the deaf and dumb people in communicating with others as knowing sign language is not something that is common to all, our project can help with their communication.

Installations:

1. numpy -> pip install numpy

2. opencv -> pip install opencv-python

3. tensorFlow -> i) pip install tensorflow 
                 ii) pip3 install --upgrade https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.8.0-cp34-cp34m-linux_x86_64.whl

4. keras -> pip install keras

5. tkinter -> pip install tk

6. PIL -> pip install Pillow

7. enchant -> pip install pyenchant 

8. hunspell -> pip install cyhunspell

Implementation of sign language project with GUI

![image](https://user-images.githubusercontent.com/88571564/167299443-f383fb95-c89f-4af9-96dd-d96089b8e779.png)

The sign language  custom file contains code for creation of dataset folders, capturing images and model creation. We used CNN for model building.

gui.py contains the code for our GUI which includes add, back, clear buttons to help the user make the sentence. We also have a button to read the whole sentence using GTTS. I have also attached the models which aren't very accurate and contain data only for A-E. you can add training and testing images by changing the mode in sign language custom file and train your own model.

