#!/usr/bin/env python
# coding: utf-8

# In[6]:


import cv2
import matplotlib.pyplot as plt
from datetime import datetime
dateTimeObj = datetime.now()

333

# In[7]:


from sklearn.cluster import KMeans
import numpy as np


# In[11]:


def convertimageactual(path,noofcolors,filename):
    image=cv2.imread(path)
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    #plt.imshow(image)
    
    orinalsize=image.shape
    flattenimage = image.reshape((-1,3))
    kmeans = KMeans(n_clusters=noofcolors)
    kmeans.fit(flattenimage)
    clustercentres = kmeans.cluster_centers_
    clustercentres = np.array(clustercentres , dtype='uint8')

    colors=[]
    for color in clustercentres:
        colors.append(color)

#     i=1
#     for each_color in clustercentres:
#     plt.subplot(1,noofcolors,i)

#     # all that is below will come in the ith subplot
#     temp = np.zeros((100,100,3),dtype='uint8')
#     temp[:,:,:] = each_color
#     plt.imshow(temp)
#     plt.axis("off")
#     i+=1

#     plt.show()

    new_image=np.zeros((flattenimage.shape[0],3),dtype='uint8')

    for ix in range(flattenimage.shape[0]):
        cluster_id = kmeans.labels_[ix]  # 0,1,2,3
        new_image[ix] = colors[cluster_id]

    new_image = new_image.reshape((orinalsize))  
#    plt.imshow(new_image)
    new_image = cv2.cvtColor(new_image,cv2.COLOR_RGB2BGR)
    
            
    new_image = cv2.cvtColor(new_image,cv2.COLOR_RGB2BGR) 
    cv2.imwrite("./static/new"+str(noofcolors)+filename,new_image)                 
#    return new_image
    

# In[12]:


def convertimage(path,noofcolors,filename):
    
    convertimageactual(path,noofcolors,filename)
     


# In[ ]:




