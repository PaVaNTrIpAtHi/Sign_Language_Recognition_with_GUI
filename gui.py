# Importing Libraries

import numpy as np
import pyttsx3
import cv2
import os, sys
import time
import operator

from string import ascii_uppercase

import tkinter as tk
from PIL import Image, ImageTk

from hunspell import Hunspell
import enchant

from keras.models import model_from_json

os.environ["THEANO_FLAGS"] = "device=cuda, assert_no_cpu_op=True"

#Application :

class Application:

    def __init__(self):
        self.word2 = ""
        self.sentence = ""

        self.hs = Hunspell('en_US')
        self.vs = cv2.VideoCapture(0)
        self.current_image = None
        self.current_image2 = None
        self.json_file = open(r"model_gui.json", "r")
        self.model_json = self.json_file.read()
        self.json_file.close()

        self.loaded_model = model_from_json(self.model_json)
        self.loaded_model.load_weights("model_gui.h5")
        


        self.ct = {}
        self.ct['blank'] = 0
        self.blank_flag = 0

        for i in ascii_uppercase:
            self.ct[i] = 0
        
        print("Loaded model from disk")

        self.root = tk.Tk()
        self.root.title("Sign Language To Text Conversion")
        self.root.protocol('WM_DELETE_WINDOW', self.destructor)
        self.root.geometry("900x1000")

        self.panel = tk.Label(self.root)
        self.panel.place(x = 100, y = 10, width = 400, height = 400)
        
        self.panel2 = tk.Label(self.root) # initialize image panel
        self.panel2.place(x = 400, y = 65, width = 300, height = 300)

        self.T = tk.Label(self.root)
        self.T.place(x = 60, y = 5)
        self.T.config(text = "Sign Language To Text Conversion", font = ("Courier", 30, "bold"))

        self.panel3 = tk.Label(self.root) # Current Symbol
        self.panel3.place(x = 420, y = 420)
        
        self.panel4 = tk.Label(self.root) # Word
        self.panel4.place(x = 220,y=460)


        self.T1 = tk.Label(self.root)
        self.T1.place(x = 10, y = 420)
        self.T1.config(text = "Character :", font = ("Courier", 30, "bold"))
        
        
        self.T = tk.Label(self.root)
        self.T.place(x = 10, y = 480)
        self.T.config(text = "Word ", font = ("Courier", 30, "bold"))

        self.panel5 = tk.Label(self.root) # Sentence
        self.panel5.place(x = 220, y = 540)

        self.T2 = tk.Label(self.root)
        self.T2.place(x = 10,y = 540)
        self.T2.config(text = "Sentence :", font = ("Courier", 30, "bold"))
        
        
        self.b = tk.Button(self.root,text = "add ", command = self.addletter, height = 0, width = 0)
        self.b.place(x = 700, y = 450)
      
        self.b1 = tk.Button(self.root, text = "space", command = self.space, height = 0, width = 0)
        self.b1.place(x = 700, y = 490)
        
        self.b11 = tk.Button(self.root,text = "backspace", command = self.delw, height = 0, width = 0)
        self.b11.place(x = 700, y = 530)

        self.b12 = tk.Button(self.root,text = "clear", command = self.clear, height = 0, width = 0)
        self.b12.place(x = 700, y = 575)
        
        #self.button = tk.Button(self.root, text="Clear",
                                    #command=lambda : self.clear2(self.T1))
        #self.button.place(x= 100 , y = 575)

        
        self.b1 = tk.Button(self.root, text = "listen ", command = self.speech, height = 0, width = 0)
        self.b1.place(x = 700, y = 620)


        self.T4 = tk.Label(self.root)
        self.T4.place(x = 250, y = 585)
        self.T4.config(text = "Suggestions :", fg = "red", font = ("Courier", 30, "bold"))

        self.bt1 = tk.Button(self.root, command = self.action1, height = 0, width = 0)
        self.bt1.place(x = 26, y = 640)

        self.bt2 = tk.Button(self.root, command = self.action2, height = 0, width = 0)
        self.bt2.place(x = 325, y = 640)

        self.bt3 = tk.Button(self.root, command = self.action3, height = 0, width = 0)
        self.bt3.place(x = 625, y = 640)


        self.str = ""
        self.word = " "
        self.current_symbol = "Empty"
        self.photo = "Empty"
        self.video_loop()


    def video_loop(self):
        ok, frame = self.vs.read()
        height,width = frame.shape[:2]

        if ok:
            cv2image = cv2.flip(frame, 1)
            #cv2image = frame


            x1 = int(0.5 * frame.shape[0])
            y1 = 10
            x2 = frame.shape[1] - 10
            y2 = int(0.5 * frame.shape[1])
            #print(x1,x2,y1,y2)

            cv2.rectangle(frame, (x1 - 1, y1 - 1), (x2 + 1, y2 + 1), (255, 0, 0) ,1)
            cv2image = cv2.cvtColor(cv2image, cv2.COLOR_BGR2RGBA)

            self.current_image = Image.fromarray(cv2image)
            imgtk = ImageTk.PhotoImage(image = self.current_image)

            self.panel.imgtk = imgtk
            self.panel.config(image = imgtk)
            
            # Extracting the ROI
            roi = cv2image[y1:y2, x1:x2]
            

            # Resizing the ROI so it can be fed to the model for prediction
            roi = cv2.resize(roi, (64, 64)) 
            roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            _, res2 = cv2.threshold(roi, 120, 255, cv2.THRESH_BINARY)
            #cv2.imshow("test", res2)


            cv2image = cv2image[y1 : y2, x1 : x2]

            gray = cv2.cvtColor(cv2image, cv2.COLOR_BGR2GRAY)

            blur = cv2.GaussianBlur(gray, (5, 5), 2)

            th3 = cv2.adaptiveThreshold(blur, 255 ,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

            ret, res = cv2.threshold(th3, 70, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            
            self.predict(res2)
            roi = cv2.resize(roi, (300, 300))
            cv2.imshow("test", roi)

            self.current_image2 = Image.fromarray(res)

            imgtk = ImageTk.PhotoImage(image = self.current_image2)

            self.panel2.imgtk = imgtk
            self.panel2.config(image = imgtk)

            self.panel3.config(text = self.current_symbol, font = ("Courier", 30))

            predicts = self.hs.suggest(self.word2)
            
            if(len(predicts) > 1):

                self.bt1.config(text = predicts[0], font = ("Courier", 20))

            else:

                self.bt1.config(text = "")

            if(len(predicts) > 2):

                self.bt2.config(text = predicts[1], font = ("Courier", 20))

            else:

                self.bt2.config(text = "")

            if(len(predicts) > 3):

                self.bt3.config(text = predicts[2], font = ("Courier", 20))

            else:

                self.bt3.config(text = "")


        self.root.after(5, self.video_loop)
      
    def predict(self,test_image):
        test_image = cv2.resize(test_image, (64, 64))

        result = self.loaded_model.predict(test_image.reshape(1, 64, 64, 1))
        prediction = {
                  'None' : result[0][0],
                  'A' : result[0][1],
                  'B' : result[0][2],
                  'C' : result[0][3],
                  'D' : result[0][4],
                  'E' : result[0][5],
                  'F' : result[0][6],
                  'G' : result[0][7],        
                  'H' : result[0][8],
                  'I' : result[0][9],
                  'J' : result[0][10],
                  'K' : result[0][11],
                  'L' : result[0][12],
                  'M' : result[0][13],
                  'N' : result[0][14],
                  'O' : result[0][15],
                  'P' : result[0][16],
                  'Q' : result[0][17],
                  'R' : result[0][18],
                  'S' : result[0][19],
                  'T' : result[0][20],
                  'U' : result[0][21],
                  'V' : result[0][22],
                  'W' : result[0][23],
                  'X' : result[0][24],
                  'Y' : result[0][25],
                  'Z' : result[0][26],
                  #'adb': result[0][27]
                 }
        sa = max(result[0]*100)
    
        # Sorting based on top prediction
        prediction = sorted(prediction.items(), key=operator.itemgetter(1), reverse=True)
        if sa>99:
            self.current_symbol = prediction[0][0]
            print(prediction[0][0])
        
    def addletter(self):
        self.word2 = self.word2 + self.current_symbol
        
        self.panel4.config(text=self.word2,font=("Courier",40))
        self.panel5.config(text=self.sentence,font=("Courier",40))
        
        
    def space(self):
        self.sentence  = self.sentence + " " + self.word2
        self.word2=""
        self.panel4.config(text=self.word2,font=("Courier",40))
        self.panel5.config(text=self.sentence,font=("Courier",40))
        
   
    def delw(self):
        self.word2 = self.word2[:-1] 
        self.panel4.config(text=self.word2,font=("Courier",40))
        
    def clear(self):
        self.word2 = ""
        self.panel4.config(text=self.word2,font=("Courier",40))
        
    def clear2(self,T):
        self.T['text'] = ""

#         self.T11 = tk.Label(self.root)
#         self.T11.place(x = 600, y = 450)
#         self.T11.config(text = self.word2, font = ("Courier", 30, "bold"))
      
    def speech(self):    
        engine = pyttsx3.init()
        engine.say(self.sentence)
        engine.runAndWait()

    def action1(self):

    	predicts = self.hs.suggest(self.word2)

    	if(len(predicts) > 0):

            #self.word = ""

            #self.word2 += " "

            self.word2 = predicts[0]
            self.panel4.config(text=self.word2,font=("Courier",40))

    def action2(self):

    	predicts = self.hs.suggest(self.word2)

    	if(len(predicts) > 1):
            #self.word = ""
            #self.word2 += " "
            self.word2 = predicts[1]
            self.panel4.config(text=self.word2,font=("Courier",40))

    def action3(self):

    	predicts = self.hs.suggest(self.word2)

    	if(len(predicts) > 2):
            #self.word = ""
            #self.word2 += " "
            self.word2 = predicts[2]
            self.panel4.config(text=self.word2,font=("Courier",40))

    def action4(self):

    	predicts = self.hs.suggest(self.word2)

    	if(len(predicts) > 3):
            #self.word = ""
            #self.word2 += " "
            self.word2 = predicts[3]
            self.panel4.config(text=self.word2,font=("Courier",40))

    def action5(self):

    	predicts = self.hs.suggest(self.word2)

    	if(len(predicts) > 4):
            #self.word = ""
            #self.word2 += " "
            self.word2 = predicts[4]
            self.panel4.config(text=self.word2,font=("Courier",40))
            
    def destructor(self):

        print("Closing Application...")

        self.root.destroy()
        self.vs.release()
        cv2.destroyAllWindows()
    
print("Starting Application...")

(Application()).root.mainloop()
