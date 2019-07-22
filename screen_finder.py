import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from scipy import optimize

import h5py
import logging

from . import plotting

class ScreenFinder:
    def __init__(self, filename):
        self.filename = filename
        self.ax = plotting.plot_screen(self.filename)
        
        self.points = [[0,0]]
        self.line, = self.ax.plot(self.points[0][0],self.points[0][1],'+')
        self.circle = Circle((0.0,0.0),0.1,color='r',fill=False)
        self.ax.add_artist(self.circle)
        self.cid = self.ax.figure.canvas.mpl_connect('button_press_event', self.draw_circle)
        self.cid2 = self.ax.figure.canvas.mpl_connect('key_press_event', self.key_press)
        
        plt.show()


    def draw_circle(self, event):
        #print('click', event)
        if event.inaxes!=self.ax: return
        self.points.append((event.xdata,event.ydata))
        
        self.line.set_data(*np.asfarray(self.points).T)
        
        #draw a circle if the number of points is enough
        if len(self.points) > 3:
            points = self.points[1:]
            radius,center = self.fit_circle(points)
            self.circle.center = center
            self.circle.radius = radius

        self.ax.figure.canvas.draw()

    def key_press(self,event):
        logging.debug('key press: {}'.format(event.key))
        if event.key == ' ':
            self.close_plot()
        
    def close_plot(self):
        #on exit write the final circle coordinates
        logging.debug(self.circle.radius)
        logging.debug(self.circle.center)
        screen_diameter = 50.038e-3
        
        with h5py.File(self.filename,'r+') as f:
            f.attrs['screen_radius'] = self.circle.radius
            f.attrs['screen_center'] = self.circle.center
            f.attrs['pixel_scale'] = screen_diameter/(self.circle.radius*2)
        plt.close(self.ax.figure)

    def calcuate_circle(self,points):
        #http://paulbourke.net/geometry/circlesphere/
    
        p1 = points[0]
        p2 = points[1]
        p3 = points[2]
        
        ma = (p2[1] - p1[1]) / (p2[0] - p1[0])
        mb = (p3[1] - p2[1]) / (p3[0] - p2[0])
        
        center_x = (ma*mb*(p1[1] - p3[1]) + mb*(p1[0] + p2[0]) - ma*(p2[0] + p3[0])) / (2 * (mb -ma))
        center_y = -(1 / ma) * (center_x - (p1[0] + p2[0]) / 2) + (p1[1] + p2[1]) / 2
        
        radius = np.sqrt((p1[0] - center_x)**2 + (p1[1] - center_y)**2)
        
        return radius,(center_x,center_y)
        
    def fit_circle(self,points):
        #https://scipy-cookbook.readthedocs.io/items/Least_Squares_Circle.html
        npts = np.asfarray(points).T
        x = npts[0]
        y = npts[1]
        x_m = np.mean(x)
        y_m = np.mean(y)
    
        method_2  = "leastsq"
        def calc_R(xc, yc):
            """ calculate the distance of each 2D points from the center (xc, yc) """
            return np.sqrt((x-xc)**2 + (y-yc)**2)


        def f_2(c):
            """ calculate the algebraic distance between the 2D points and the mean circle centered at c=(xc, yc) """
            Ri = calc_R(*c)
            return Ri - Ri.mean()

        center_estimate = x_m, y_m
        center_2, ier = optimize.leastsq(f_2, center_estimate)

        xc_2, yc_2 = center_2
        Ri_2       = calc_R(xc_2, yc_2)
        R_2        = Ri_2.mean()
        residu_2   = sum((Ri_2 - R_2)**2)
        residu2_2  = sum((Ri_2**2-R_2**2)**2)

        return R_2,(xc_2,yc_2)

    def get_pixel_scale(self):
        return self.screen_diameter / self.circle.radius * 2
        
if __name__=='__main__':
    logging.basicConfig(level=logging.DEBUG)
    for i in range(3):
        screenFinder = ScreenFinder('10_17/EYG2_{}.dat'.format(i))

    logging.debug('done')
