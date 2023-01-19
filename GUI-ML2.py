# -*- coding: utf-8 -*-
"""
Created on Tue May  5 10:50:31 2020

@author: laksh
"""

#from tkinter import *  
#creating the application main window.   
#top = Tk()  
#Entering the event main loop  
#top.mainloop()  
import tkinter as tk
from tkinter import ttk
import numpy as np
import pickle

s=[]
p=np.array(6)

# load the model from disk
filename = 'G:/MYResearch/Covid-Data-Set/PhC-DataSet/DataSetCreation-Code/GUI/finalized_model.sav'
loaded_model = pickle.load(open(filename, 'rb'))


def click_me():
    s.append(e1v.get())
    s.append(e2v.get())
    s.append(e3v.get())
    s.append(e4v.get())
    s.append(e5v.get())
    s.append(e6v.get())
    p=np.array(s).astype(int)
    print(p)
    res=loaded_model.predict([[p[0],p[1],p[2],p[3],p[4],p[5]]])
    if res[0]=='HR1':
        ttk.Label(top,text="The Result is"+"  "+"HIGH-RISK").place(x = 30, y = 400)
        ttk.Label(top,text="COVID-19 Test is Must").place(x = 30, y = 430)
    elif res[0]=='HR2':
        ttk.Label(top,text="The Result is"+"  "+"MEDIUM-RISK").place(x = 30, y = 400)
        ttk.Label(top,text="COVID-19 Test is required").place(x = 30, y = 430)
    elif res[0]=='HR3':
        ttk.Label(top,text="The Result is"+"  "+"LOW-RISK").place(x = 30, y = 400)
        ttk.Label(top,text="COVID-19 Test is Recommended").place(x = 30, y = 430)
    elif res[0]=='GS':
         ttk.Label(top,text="The Result is"+"  "+str(res[0])).place(x = 30, y = 400)
         ttk.Label(top,text="It is General Symptom and not").place(x = 30, y = 430)
         ttk.Label(top,text="recommended for COVID-19 Test").place(x = 30, y = 450)

top = tk.Tk()  
top.geometry("385x500")
top.title("COVID-19 Test Recommendation System")

e1v = tk.StringVar()
e2v = tk.StringVar()
e3v = tk.StringVar()
e4v = tk.StringVar()
e5v = tk.StringVar()
e6v = tk.StringVar()  



lbl = ttk.Label(top, text = "Symptoms").place(x = 30,y = 10)
Cold = ttk.Label(top, text = "Cold").place(x = 30,y = 50)
Cough = ttk.Label(top, text = "Cough").place(x = 30, y = 90)
Feaver = ttk.Label(top, text = "Feaver").place(x = 30, y = 130) 
Breating = ttk.Label(top, text = "Breating").place(x = 30, y = 170)
lbl1 = ttk.Label(top, text = "Other Information").place(x = 30,y = 210)
Chronic = ttk.Label(top, text = "Chronic").place(x = 30, y = 250) 
Age = ttk.Label(top, text = "Age").place(x = 30, y = 290)
sbmitbtn = ttk.Button(top, text = "Submit",command = click_me).place(x = 30, y = 330)



lbl2 = ttk.Label(top)
lbl2.place(x = 30,y = 450) 

e1 = ttk.Entry(top,textvariable=e1v).place(x = 95, y = 50)  
e2 = ttk.Entry(top,textvariable=e2v).place(x = 95, y = 90)  
e3 = ttk.Entry(top,textvariable=e3v).place(x = 95, y = 130)
e4 = ttk.Entry(top,textvariable=e4v).place(x = 95, y = 170)

e5 = ttk.Entry(top,textvariable=e5v).place(x = 95, y = 250)
e6 = ttk.Entry(top,textvariable=e6v).place(x = 95, y = 290)

label = ttk.Label(top)  
label.pack() 
top.mainloop()  