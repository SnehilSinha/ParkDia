from tkinter import *
import cv2
from keras.models import load_model
from time import sleep
from keras.preprocessing.image import img_to_array
from keras.preprocessing import image
import numpy as np
#import parselmouth
import matplotlib.pyplot as plt
import seaborn as sns


class Page(Frame):
    def __init__(self, *args, **kwargs):
        Frame.__init__(self, *args, **kwargs)
    def show(self):
        self.lift()

class Page1(Page):
   
   def __init__(self, *args, **kwargs):
        Page.__init__(self, *args, **kwargs)
        pname=StringVar()
        pgen=StringVar()
        page=StringVar()
        pinh=StringVar()
        psleep=StringVar()
        l1 = Label( self,text="Fill the following information of the patient: ")
        l2 = Label( self, text="Patient ID",width=20,font=("bold", 10))
        l2.place(x=60,y=130)
        e2 = Entry( self, textvar=pname, width=30)
        e2.place(x=280,y=130)
        l3 = Label( self, text="Gender",width=20,font=("bold", 10))
        l3.place(x=60,y=200)
        e3 = Entry( self, textvar=pgen, width=30)
        e3.place(x=280,y=200)
        l4 = Label( self, text="Age",width=20,font=("bold", 10))
        l4.place(x=60,y=270)
        e4 = Entry( self, textvar=page, width=30)
        e4.place(x=800,y=270)
        l5 = Label( self, text="Disease occuring prior in Family?",width=40,font=("bold", 10))
        l5.place(x=60,y=340)
        e5 = Entry( self, textvar=pinh, width=30)
        e5.place(x=800,y=340)
        l6 = Label( self, text="Sleep deprived/unusual phase?",width=40,font=("bold", 10))
        l6.place(x=60,y=410)
        e6 = Entry( self, textvar=psleep, width=30)
        e6.place(x=800,y=410)
        self.quit()
        

class Page2(Page):
   def __init__(self, *args, **kwargs):
    Page.__init__(self, *args, **kwargs)
    def snapsnap():
            key = cv2. waitKey(1)
            webcam = cv2.VideoCapture(0)
            while True:
                try:
                    check, frame = webcam.read()
                    print(check) #prints true as long as the webcam is running
                    print(frame) #prints matrix values of each framecd 
                    cv2.imshow("Capturing", frame)
                    key = cv2.waitKey(1)
                    if key == ord('s'): 
                        cv2.imwrite(filename='saved_img.jpg', img=frame)
                        webcam.release()
                        img_new = cv2.imread('saved_img.jpg', cv2.IMREAD_GRAYSCALE)
                        img_new = cv2.imshow("Captured Image", img_new)
                        cv2.waitKey(1650)
                        cv2.destroyAllWindows()
                        print("Processing image...")
                        img_ = cv2.imread('saved_img.jpg', cv2.IMREAD_ANYCOLOR)
                        print("Converting RGB image to grayscale...")
                        gray = cv2.cvtColor(img_, cv2.COLOR_BGR2GRAY)
                        print("Converted RGB image to grayscale...")
                        print("Resizing image to 28x28 scale...")
                        img_ = cv2.resize(gray,(28,28))
                        print("Resized...")
                        img_resized = cv2.imwrite(filename='saved_img-final.jpg', img=img_)
                        print("Image saved!")
        
                        break
            
                    elif key == ord('q'):
                        print("Turning off camera.")
                        webcam.release()
                        print("Camera off.")
                        print("Program ended.")
                        cv2.destroyAllWindows()
                        break
        
                except(KeyboardInterrupt):
                    print("Turning off camera.")
                    webcam.release()
                    print("Camera off.")
                    print("Program ended.")
                    cv2.destroyAllWindows()
                    break
                
            
    b1 = Button(self, text="Upload Spiral Image", width= 20, anchor="center",command=snapsnap)
    b1.place(x=180,y=200)

class Page3(Page):
   def __init__(self, *args, **kwargs):
    Page.__init__(self, *args, **kwargs)
    def snapsnap():
            key = cv2. waitKey(1)
            webcam = cv2.VideoCapture(0)
            while True:
                try:
                    check, frame = webcam.read()
                    print(check) #prints true as long as the webcam is running
                    print(frame) #prints matrix values of each framecd 
                    cv2.imshow("Capturing", frame)
                    key = cv2.waitKey(1)
                    if key == ord('s'): 
                        cv2.imwrite(filename='saved_img.jpg', img=frame)
                        webcam.release()
                        img_new = cv2.imread('saved_img.jpg', cv2.IMREAD_GRAYSCALE)
                        img_new = cv2.imshow("Captured Image", img_new)
                        cv2.waitKey(1650)
                        cv2.destroyAllWindows()
                        print("Processing image...")
                        img_ = cv2.imread('saved_img.jpg', cv2.IMREAD_ANYCOLOR)
                        print("Converting RGB image to grayscale...")
                        gray = cv2.cvtColor(img_, cv2.COLOR_BGR2GRAY)
                        print("Converted RGB image to grayscale...")
                        print("Resizing image to 28x28 scale...")
                        img_ = cv2.resize(gray,(28,28))
                        print("Resized...")
                        img_resized = cv2.imwrite(filename='saved_img-final.jpg', img=img_)
                        print("Image saved!")
        
                        break
            
                    elif key == ord('q'):
                        print("Turning off camera.")
                        webcam.release()
                        print("Camera off.")
                        print("Program ended.")
                        cv2.destroyAllWindows()
                        break
        
                except(KeyboardInterrupt):
                    print("Turning off camera.")
                    webcam.release()
                    print("Camera off.")
                    print("Program ended.")
                    cv2.destroyAllWindows()
                    break
                
            
    b1 = Button(self, text="Upload Wave Image", width= 20, anchor="center",command=snapsnap)
    b1.place(x=180,y=200)

class Page4(Page):
       def __init__(self, *args, **kwargs):
        Page.__init__(self, *args, **kwargs)
        def facRig():
            face_classifier = cv2.CascadeClassifier('F:\Projects\HorusWeb\haarcascade_frontalface_default.xml')
            classifier =load_model('F:\Projects\HorusWeb\Emotion_little_vgg.h5')
            class_labels = ['Healthy','Healthy','Rigid Facial Muscles','Healthy','Healthy']
            cap = cv2.VideoCapture(0)
            while True:
        # Grab a single frame of video
                ret, frame = cap.read()
                labels = []
                gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
                faces = face_classifier.detectMultiScale(gray,1.3,5)

                for (x,y,w,h) in faces:
                    cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
                    roi_gray = gray[y:y+h,x:x+w]
                    roi_gray = cv2.resize(roi_gray,(48,48),interpolation=cv2.INTER_AREA)
        # rect,face,image = face_detector(frame)


                    if np.sum([roi_gray])!=0:
                        roi = roi_gray.astype('float')/255.0
                        roi = img_to_array(roi)
                        roi = np.expand_dims(roi,axis=0)
                        sumation=0
        # logan=0

            # make a prediction on the ROI, then lookup the class

                        preds = classifier.predict(roi)[0]
                        label=class_labels[preds.argmax()]
                        if(label =="Healthy"):
                            sumation=1
            #print(sumation)	
                        
                        label_position = (x,y)
                        cv2.putText(frame,label,label_position,cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),3)
                    else:
                            cv2.putText(frame,'No Face Found',(20,60),cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),3)
                cv2.imshow('Emotion Detector',frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            cap.release()
            cv2.destroyAllWindows()
        
        b1 = Button(self, text="Take face rigidity Test", width= 20, anchor="center",command=facRig)
        b1.place(x=180,y=200)

"""class Page5(Page):
       def __init__(self, *args, **kwargs):
            Page.__init__(self, *args, **kwargs)
            sns.set() # Use seaborn's default style to make attractive graphs
            plt.rcParams['figure.dpi'] = 100 # Plot nice figures using Python's "standard" matplotlib library
            snd = parselmouth.Sound("C:/Users/snehi/OneDrive/Documents/PD SIH/asdf.wav3")
            plt.figure()
            plt.plot(snd.xs(), snd.values.T)
            plt.xlim([snd.xmin, snd.xmax])
            plt.xlabel("time [s]")
            plt.ylabel("amplitude [Pa]")
            plt.show() # or plt.savefig("sound.png"), or plt.savefig("sound.pdf")
            t2 = len(snd.values.T)
            print(t2,"seconds")


            if(t1> t2):
                print("Patient shows symptoms of Parkinson's")
            else:
                print("Normal Patient")
"""
class MainView(Frame):
    def __init__(self, *args, **kwargs):
        Frame.__init__(self, *args, **kwargs)
        p1 = Page1(self)
        p2 = Page2(self)
        p3 = Page3(self)
        p4 = Page4(self)
        #p5 = Page5(self)

        buttonframe = Frame(self)
        container = Frame(self)
        buttonframe.pack(side="top", fill="x", expand=False)
        container.pack(side="top", fill="both", expand=True)

        p1.place(in_=container, x=0, y=0, relwidth=1, relheight=1)
        p2.place(in_=container, x=0, y=0, relwidth=1, relheight=1)
        p3.place(in_=container, x=0, y=0, relwidth=1, relheight=1)
        p4.place(in_=container, x=0, y=0, relwidth=1, relheight=1)
        #p5.place(in_=container, x=0, y=0, relwidth=1, relheight=1)

        b1 = Button(buttonframe, text="Patient", command=p1.lift)
        b2 = Button(buttonframe, text="Spiral", command=p2.lift)
        b3 = Button(buttonframe, text="Wave", command=p3.lift)
        b4 = Button(buttonframe, text="Rigidity", command=p4.lift)
        #b5 = Button(buttonframe, text="Rigidity", command=p4.lift)

        b1.pack(side="left")
        b2.pack(side="left")
        b3.pack(side="left")
        b4.pack(side="left")
        #b5.pack(side="left")

        p1.show()

if __name__ == "__main__":
    root = Tk()
    main = MainView(root)
    main.pack(side="top", fill="both", expand=True)
    root.wm_geometry("500x500")
    
    root.mainloop()
