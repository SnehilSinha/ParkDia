#!/usr/bin/env python
# coding: utf-8

# In[7]:


pip install praat-parselmouth


# In[61]:


#To find out if patient is speaking softly
import parselmouth

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set() # Use seaborn's default style to make attractive graphs

plt.rcParams['figure.dpi'] = 100 # Plot nice figures using Python's "standard" matplotlib library
snd = parselmouth.Sound("C:/Users/snehi/OneDrive/Documents/PD SIH/untitled.wav")
plt.figure()
plt.plot(snd.xs(), snd.values.T)
plt.xlim([snd.xmin, snd.xmax])
plt.xlabel("time [s]")
plt.ylabel("amplitude")
plt.show() # or plt.savefig("sound.png"), or plt.savefig("sound.pdf")

thresh = 0.0009
#print(len(snd.xs()))
flag = 0
for i in snd.values.T:
    if(i>thresh):
        flag = flag +1
#print(flag)

if((len(snd.xs()))/flag > 0.5):
    print("Normal Patient speaking")
else:
    print("Patient shows symptoms of Parkinson's")
    
    


# In[66]:

#t To find out patient is speaking slowly, taking time to think
import parselmouth

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set() # Use seaborn's default style to make attractive graphs

 # Plot nice figures using Python's "standard" matplotlib library
snd = parselmouth.Sound("C:/Users/snehi/OneDrive/Documents/PD SIH/123.wav")
plt.figure()
plt.plot(snd.xs(), snd.values.T)
plt.xlim([snd.xmin, snd.xmax])
plt.xlabel("time [s]")
plt.ylabel("amplitude [Pa]")
plt.show() # or plt.savefig("sound.png"), or plt.savefig("sound.pdf")
t1 = len(snd.values.T)
print(t1,"seconds")


snd = parselmouth.Sound("C:/Users/snehi/OneDrive/Documents/PD SIH/asdf.wav")
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
    
    


# In[ ]:





# In[ ]:




