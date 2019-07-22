import numpy as np
import matplotlib.pyplot as plt


#class for selecting and isolating parts of an image array using a variety of shapes - mainly to select a background to remove objects in frame
#create background image by using numpy masking in order to select/deselect parts of the array

class Shape:
    def __init__(self):
        pass

class Rectangle(Shape):
    def __init__(self,center,x_size,y_size):
        Shape.__init__(self)
        self.center = center
        self.x_size = x_size
        self.y_size = y_size

    def inside_shape(self,coords):
        return (abs(coords[0] - self.center[0]) < self.x_size/2) and (abs(coords[1] - self.center[1]) < self.y_size/2)

class Circle(Shape):
    def __init__(self,center,radius):
        Shape.__init__(self)
        self.center = center
        self.radius = radius

    def inside_shape(self,coords):
        return np.sqrt((coords[1]-self.center[0])**2 + (coords[0] - self.center[1])**2) < self.radius

def create_background(image_array,shapes):
    background = np.ones_like(image_array)*4000
    image_shape = image_array.shape
    
    for shape in shapes:
        for i in range(image_shape[1]):
            for j in range(image_shape[0]):
                if shape.inside_shape([j,i]):
                    background[j][i] = image_array[j][i]

    return background

def subtract_background(image,background):
    new_image = np.asarray(image,dtype = np.int32)
    new_image -= np.asarray(background,dtype = np.int32) 
    new_image = np.where(new_image >= 0,new_image,0)
    return np.asarray(new_image,dtype = np.uint16)

def histogram_frame(data):
    #data = data.flatten()
    h,be = np.histogram(data,bins=100)#,range=(0,9000))
    bc = (be[1:] + be[:-1])/2
    fig,ax = plt.subplots()
    ax.plot(bc,h)


