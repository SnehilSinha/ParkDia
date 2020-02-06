from Tkinter import *
import os
import subprocess

def test_sp():
    mo=Toplevel(root)
    mo.configure(background="#FFFFFF")
    mo.title("Spiral Test")
    
    upload=Button(mo,text="upload image",height=2,width=30,bg = "#B3D8DB",font = ("TIMES NEW ROMAN",15))
    upload.place(x =100,y=50)

    mo.geometry("500x500")
    mo.mainloop()



def test_bal():
    t=Toplevel(root)
    t.configure(background="#FFFFFF")
    t.title("Wave Test")
   
    openc=Button(t,text="upload image",height=2,width=30,bg = "#b3d8db",font = ("TIMES NEW ROMAN",15))
    openc.place(x=100 , y=50)


    t.geometry("500x500")
    t.mainloop()
    
def test_face():
    t=Toplevel(root)
    t.configure(background="#FFFFFF")
    t.title("facial regidity")
   
    openc=Button(t,text="Open Camera",height=2,width=30,bg = "#b3d8db",font = ("TIMES NEW ROMAN",15))
    openc.place(x=100 , y=50)


    t.geometry("500x500")
    t.mainloop()

def main_screen():
    global root
    root=Tk()
    root.geometry("1500x1500")
    root.configure(background="#ECECEC")
    root.title("Horus")
    title = Label(root, text="Parkinson Detector" ,bg="#C4D6D8",width=400,height=3,font = ("TIMES NEW ROMAN",20))
    title.pack()
    
    loginuser=Button(root,text="Test spiral",height=2,width=30,bg = "#B3D8DB",font = ("TIMES NEW ROMAN",15),command=test_sp)
    loginuser.place(x=750 , y=250)
    
    loginowner=Button(root,text="Test wave",height=2,width=30,bg = "#B3D8DB",font = ("TIMES NEW ROMAN",15),command=test_bal)
    loginowner.place(x=750, y=150)
    
    face=Button(root,text="facial regidity",height=2,width=30,bg = "#B3D8DB",font = ("TIMES NEW ROMAN",15),command=test_face)
    face.place(x=750, y=350)

    root.mainloop()

main_screen()

