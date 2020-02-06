from tkinter import *

def test_hw():
    mo=Toplevel(root)
    mo.configure(background="light coral")
    mo.title("Test Hand Writing")
    welcome = Label(mo, text='Welcome ',bg='Blue',font = ("TIMES NEW ROMAN",20))
    welcome.pack()
    upload=Button(mo,text="Upload Picture of drawing",height=2,width=30,bg = "red4",font = ("TIMES NEW ROMAN",15))
    upload.pack()


    mo.geometry("500x500")
    mo.mainloop()



def test_bal():
    t=Toplevel(root)
    t.configure(background="light coral")
    t.title("Test Balance")
    welcome = Label(t, text='Welcome ',bg='Blue',font = ("TIMES NEW ROMAN",20))
    welcome.pack()
    openc=Button(t,text="Open Camera",height=2,width=30,bg = "red4",font = ("TIMES NEW ROMAN",15))
    openc.pack()


    t.geometry("500x500")
    t.mainloop()
    

def main_screen():
    global root
    root=Tk()
    root.geometry("1500x1500")
    root.configure(background="black")
    root.title("Horus")
    title = Label(root, text=' Parkinson Detector ',bg='maroon4',width=400,height=3,font = ("TIMES NEW ROMAN",20))
    title.pack()
    loginuser=Button(root,text="Test Handwriting",height=2,width=30,bg = "red4",font = ("TIMES NEW ROMAN",15),command=test_hw)
    loginuser.pack()
    loginowner=Button(root,text="Test Balance",height=2,width=30,bg = "red4",font = ("TIMES NEW ROMAN",15),command=test_bal)
    loginowner.pack()

    root.mainloop()

main_screen()
