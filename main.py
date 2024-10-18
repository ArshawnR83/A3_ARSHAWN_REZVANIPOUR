# -*- coding: utf-8 -*-
"""
Created on Tue Oct 14 08:30:48 2024

@author: arshawn
"""




#Classification

#breast_cancer

print('1.What is the definition of breast cancer? Breast cancer is a disease in which abnormal breast cells grow out of control and form tumours. If left unchecked, the tumours can spread throughout the body and become fatal. Breast cancer cells begin inside the milk ducts and/or the milk-producing lobules of the breast. 2.How is breast cancer caused? The most common genetic mutations involve the BRCA1 and BRCA2 genes. Smoking: Tobacco use has been linked to many different types of cancer, including breast cancer. Drinking beverages containing alcohol: Research shows that drinking beverages containing alcohol may increase breast cancer risk. Having obesity. 3.What is the most common age for breast cancer? Breast cancer in women Rates begin to increase after age 40 and are highest in women over age 70 (see Figure 2.1 below). Breast cancer mainly occurs in middle-aged and older women. The median age at the time of breast cancer diagnosis is 62. This means half of the women who developed breast cancer are 62 years of age or younger when they are diagnosed. A very small number of women diagnosed with breast cancer are younger than 45. 4.is breast cancer treatable? Breast cancer is highly treatable, and the majority of patients who receive proper care live long, healthy lives.')

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
data=load_breast_cancer()
import pandas as pd

a=np.random.uniform((30),(569),size=(30,3))
data=pd.DataFrame(a,index=['mean radius', 'mean texture', 'mean perimeter', 'mean area',
       'mean smoothness', 'mean compactness', 'mean concavity',
       'mean concave points', 'mean symmetry', 'mean fractal dimension',
       'radius error', 'texture error', 'perimeter error', 'area error',
       'smoothness error', 'compactness error', 'concavity error',
       'concave points error', 'symmetry error',
       'fractal dimension error', 'worst radius', 'worst texture',
       'worst perimeter', 'worst area', 'worst smoothness',
       'worst compactness', 'worst concavity', 'worst concave points',
       'worst symmetry', 'worst fractal dimension'],columns=['age between 20 to 45','age between 50 to 65','age between 70 to 90'])

data.info()

fig=plt.figure()
a=plt.axes(projection='3d')
zline=np.linspace(0,15,1000)
xline=np.sin(zline)
yline=np.cos(zline)
a.plot3D(xline,yline,zline,'blue')
plt.xlabel('All patients')
plt.ylabel(' all symptoms')
plt.show()
zdata=30*np.random.random(1000)
xdata=np.sin(zdata)+0.1*np.random.randn(1000)
ydata=np.cos(zdata)+0.1*np.random.randn(1000)
a.scatter3D(xdata,ydata,zdata,c=zdata,cmap='pink');

print('result of breast cancer in the range of 60 years: ',data)
print('=====> advancements in diagnosis and highly individualized treatment plans are increasing the odds of recovery for older patients and making it possible for many to live longer, healthier lives.')



