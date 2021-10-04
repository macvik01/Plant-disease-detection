import cv2 #open CV module
import matplotlib # mathlab num and all that
import numpy as np # for working with array like transform to matries,foriers et cetera
import scipy # to perform some mathematical calculations visualize data on large and wide scale
import scipy.io as sio
import imutils # processing functions such as translation, rotation, resizing, skeletonization, and displaying
import os # to modify a directory
import mahotas as mt # computer vision and image processing library and increases the speed of operation in numpy arrays
from imutils import contours
from sklearn.cluster import KMeans
from sklearn.cluster import spectral_clustering #->means to identify nodes in a graph based edges connecting them
from sklearn.neural_network import MLPClassifier #  multilayer perceptron form of neural network amd cam
                                                 # can distinguish data that is not linearly separable.
import csv # that .csv file something like spreadsheet and all.


# Loading image
os.chdir(r'C:\example\Our final project -all in one folder\leaffinal\leaf\test')   #change here the test image path(here its my path)
# In the above line just include the location(path) of the image 
img = cv2.imread('T1.jpg')#upload any image in the *(test) folder
print(img)
os.chdir(r'C:\MY FILES\Our final project -all in one folder\leaffinal\leaf') # hange here to original code path
cv2.imshow('Input Image',img)

##Gaussian Blurring

kernel = np.ones((7,7),np.float32)/25# M3 paper i think so and gauss formula 1/25*matrix
img1= cv2.GaussianBlur(img,(5,5),0)# accuracy of image(550) and also gauss Blurring
print(img1)

#img1 = cv2.filter2D(img,-1,kernel) normal 3x3 averaging
cv2.imshow('Gaussian Image',img1)

## Bilateral Filter for Edge Enhancement
img3 = cv2.bilateralFilter(img1,9,75,75) # 9,75,75 are the intensity of the image like resolution ,height ,width
#print(img3)
cv2.imshow('Bilateral Filtered Image',img3)


## RGB to Gray conversion
GRAY_Img = cv2.cvtColor(img3,cv2.COLOR_BGR2GRAY)
cv2.imshow('GRAY Image',GRAY_Img)
#print(GRAY_Img)

Data2Ext=GRAY_Img # temp Data2Ext
cv2.imwrite('ImageRedist.jpg',Data2Ext) # like creating
print( "\nPERMAN--End of blur,bifilter and gray--\n")# Making sure that all the above executed perfectly

roi1=GRAY_Img # temp roi
r,c=roi1.shape # .shpae gives tuple with no of elements per axis
p=1 # to make sure the final matches the initial while reshaping the image
if p==1:
    # Region of image =roi
    roi = roi1.reshape((roi1.shape[0] * roi1.shape[1], 1)) # reshaping w.r.t to original

## KMEANS clustering
imgkmeans = KMeans(n_clusters=2, random_state=0) # total cluster is 2 and nothing is in random state
imgkmeans.fit(roi)  # estimates the best representative function..
                    # like .fit  rearranges and makes necessary changes for better accuracy

label_values=imgkmeans.labels_ #specifies where  we can place the the image
Label_reshped = np.reshape(label_values,(roi1.shape[0] ,roi1.shape[1])) # reshaping the array


segmentregions=roi1
blobregions=roi1

rows,cols = roi1.shape
# Thresholding for segmentation i.e black and white a image
for i in range(0,rows):
    #since Thresholding means B&W so we must convert pixels to B&W
    for j in range(0,cols):
        pixl=Label_reshped[i,j]
        if pixl==0:
            segmentregions[i,j]=255 # white is 255 in Red and blue

        else:
            segmentregions[i,j]=0 # black is 0 in R&B

cv2.imshow('Segemented Image',segmentregions)
#contours in an image has some relationship to each other and  Representation of this relationship is called the Hierarchy.
contours, hierarchy = cv2.findContours(segmentregions, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2:]
#Since the hierarchy will come in reverse last 2 we need to get them right
# RETR_TREE is MR.Perfect like it retrieves all the contours and creates a full family hierarchy list
#CHAIN_APPROX SIMPLE means it removes all redundant points and compresses the contour, thereby saving memory.
contour_list = []
for contour in contours:
    area = cv2.contourArea(contour)
    print(area) # checking contour areas

    if area > 100 :
        contour_list.append(contour)

cnt = contours[1:]
cv2.drawContours(segmentregions, contour_list,  -1, (255,0,0), 2)
cv2.imshow('Regions Detected',segmentregions)
#print (contour_list) checking the list
print( "\n--contours and hierarchy--PERMAN \n")
#To do GLCM(Gray-Level Co-Occurrence Matrix) we need the original image to grey scale and so comes the next steps
# Thresholding for segmentation
NewImage = cv2.imread('ImageRedist.jpg')
NewImage= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
extractedregions=NewImage
#similar to above steps
for k in range(0,rows):
    for l in range(0,cols):
        pixl1=segmentregions[k,l]
        if pixl1==0:
#            print ('ok')
            extractedregions[k,l]=NewImage[k,l]
#           print (extractedregions[k,l])
        else:
#            print ('no')
            extractedregions[k,l]=0

cv2.imshow('Extracted Regions Image',extractedregions)

#A GLCM is a histogram of co-occurring greyscale values at a given offset over an image.
## GLCM Features Extractor
def extract_features(image):
        # calculate haralick texture features for 4 types of adjacency ,haralick is texture based classifion
        #Total 14 textures but 13 are used for computing
        textures = mt.features.haralick(image)

        # take the mean of it and return it as
        ht_mean = textures.mean(axis=0)
        return ht_mean


GLCMfeatures = extract_features(extractedregions)
(means, stds) = cv2.meanStdDev(extractedregions) # putting them in Tuples
## Normal Mean Standard Deviation
print( GLCMfeatures)
print (means, stds)

al=np.size(GLCMfeatures) #size  count the number of elements along a given axis
#print(al)

Id=np.zeros((al+2,), dtype=float)
#print(Id)
#dtype means data type and valuves are usually int int /float sometimes python objects .
for i in range(0,al+2):
    #print (i)
    #Normally, the feature vector is taken to be of 13-dim as computing 14th dim might increase the computational time.
    if i<13: # haralick textures total 14 check print statement.
        Id[i]=GLCMfeatures[i]
    elif i==13:
        Id[i]=means
    else:
        Id[i]=stds
#print(Id)
valuu=np.mean(Id)
print(valuu)
print (np.mean(Id))
'''
Under develepoment
p=1
W=np.zeros((15), dtype=float)
DBVal=np.zeros((20,), dtype=float)
print(W)
print(DBVal)
'''
#waitkey is keyboard that thing milliseconds after clicking and desetroy all destroyAllWindows is destory
cv2.waitKey(0)
cv2.destroyAllWindows()
