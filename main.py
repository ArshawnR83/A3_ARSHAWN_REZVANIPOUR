# -*- coding: utf-8 -*-
"""
Created on Tue Oct 14 08:30:48 2024

@author: arshawn
"""




#Classification
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
data=load_breast_cancer()

x=data.data
y=data.target

import pandas as pd
data=pd.DataFrame([[0,1,2,3,4,5,6,7,8,9,10,11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29],[30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59],[60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89],[90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 
100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 
110, 111, 112, 113, 114, 115,116,117,118,119],[120,121,122,123,124,125,126,127,128,129,130,131,132,133,134,135,136,137,138,139,140,141,142,143,144,145,146,147,148,149],[150,151,152,153,154,155,156,157,158,159,160,161,162,163,164,165,166,167,168,169,170,171,172,173,174,175,176,177,178,179],[180,181,182,183,184,185,186,187,188,189,190,191,192,193,194,195,196,197,198,199,200,201, 202, 203, 204, 205, 206, 207, 208, 209],[ 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239],[240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269],[270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299],[300,301, 302, 303, 304, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 325, 326, 327, 328, 329],[ 330, 331, 332, 333, 334, 335, 336, 337, 338, 339, 340, 341, 342, 343, 344, 345, 346, 347, 348, 349, 350, 351, 352, 353, 354, 355, 356, 357, 358, 359],[360, 361, 362, 363, 364, 365, 366, 367, 368, 369, 370, 371, 372, 373, 374, 375, 376, 377, 378, 379, 380, 381, 382, 383, 384, 385, 386, 387, 388, 389],[390, 391, 392, 393, 394, 395, 396, 397, 398, 399, 400,401, 402, 403, 404, 405, 406, 407, 408, 409, 410, 411, 412, 413, 414, 415, 416, 417, 418, 419],[420, 421, 422, 423, 424, 425, 426, 427, 428, 429, 430, 431, 432, 433, 434, 435, 436, 437, 438, 439, 440, 441, 442, 443, 444, 445, 446, 447, 448, 449],[ 450, 451, 452, 453, 454, 455, 456, 457, 458, 459, 460, 461, 462, 463, 464, 465, 466, 467, 468, 469, 470, 471, 472, 473, 474, 475, 476, 477, 478, 479],[480, 481, 482, 483, 484, 485, 486, 487, 488, 489, 490, 491, 492, 493, 494, 495, 496, 497, 498, 499, 500,510,520,530,540,550,560,565,566,569]],
                  columns=['mean radius', 'mean texture', 'mean perimeter', 'mean area',
                         'mean smoothness', 'mean compactness', 'mean concavity',
                         'mean concave points', 'mean symmetry', 'mean fractal dimension',
                         'radius error', 'texture error', 'perimeter error', 'area error',
                         'smoothness error', 'compactness error', 'concavity error',
                         'concave points error', 'symmetry error',
                         'fractal dimension error', 'worst radius', 'worst texture',
                         'worst perimeter', 'worst area', 'worst smoothness',
                         'worst compactness', 'worst concavity', 'worst concave points',
                         'worst symmetry', 'worst fractal dimension'])


print('1.What is the definition of breast cancer? Breast cancer is a disease in which abnormal breast cells grow out of control and form tumours. If left unchecked, the tumours can spread throughout the body and become fatal. Breast cancer cells begin inside the milk ducts and/or the milk-producing lobules of the breast. 2.How is breast cancer caused? The most common genetic mutations involve the BRCA1 and BRCA2 genes. Smoking: Tobacco use has been linked to many different types of cancer, including breast cancer. Drinking beverages containing alcohol: Research shows that drinking beverages containing alcohol may increase breast cancer risk. Having obesity. 3.What is the most common age for breast cancer? Breast cancer in women Rates begin to increase after age 40 and are highest in women over age 70 (see Figure 2.1 below). Breast cancer mainly occurs in middle-aged and older women. The median age at the time of breast cancer diagnosis is 62. This means half of the women who developed breast cancer are 62 years of age or younger when they are diagnosed. A very small number of women diagnosed with breast cancer are younger than 45. 4.is breast cancer treatable? Breast cancer is highly treatable, and the majority of patients who receive proper care live long, healthy lives.')
data.info()

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=560, shuffle=True, random_state=30)

from sklearn.svm import SVC

model=SVC()

model=SVC(kernel='poly',degree=2)

model.fit(x_train,y_train)

y_train_pred=model.predict(x_train)


fig = plt.figure()
ax = plt.axes(projection='3d')



zline = np.linspace(0, 15, 1000)
xline = np.sin(zline)
yline = np.cos(zline)
ax.plot3D(xline, yline, zline, 'purple')


zdata = 15 * np.random.random(100)
xdata = np.sin(zdata) + 0.1975 * np.random.randn(100)
ydata = np.cos(zdata) + 0.2015 * np.random.randn(100)
ax.scatter3D(xdata, ydata, zdata, c=zdata, cmap='pink');

from sklearn.metrics import accuracy_score
train_score=accuracy_score(y_train, y_train_pred)
print(' dtype=<U9: ',train_score)

y_test_pred=model.predict(x_test)
test_score=accuracy_score(y_test,y_test_pred)
print('result of breast cancer in the range of 60 years: ',test_score)
print('=====> advancements in diagnosis and highly individualized treatment plans are increasing the odds of recovery for older patients and making it possible for many to live longer, healthier lives.')








