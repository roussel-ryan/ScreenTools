import numpy as np
import logging
from . import current
from . import plasma

#function class for creating and storing constraints used for data processing - to be used with utils.get_frames

class Constraint(object):
    '''
    used as a container for a <function> which takes the form
    of <function>(h5py.File,frame_number,*args,**kwargs) which must
    return a boolean depending on if the particular frame 
    satisfies the constraint, if it does return True 
    
    '''
    def __init__(self,function,name,*args,**kwargs):
        self.function = function
        self.args = args
        self.kwargs = kwargs
        self.name = name

    def evaluate(self,filename,frame_number):
        return self.function(filename,frame_number,*self.args,**self.kwargs)

def null_function(f,i):
    return True

#specific constraints
class NullConstraint(Constraint):
    def __init__(self):
        Constraint.__init__(self,null_function,'NullConstraint')
        
class MeanChargeConstraint(Constraint):
    def __init__(self):
        Constraint.__init__(self,current.mean_charge,'MeanChargeConstraint')

class SelectChargeConstraint(Constraint):
    def __init__(self,lower_limit,upper_limit,ICT_channel=0):
        Constraint.__init__(self,current.selected_charge,'SelectChargeConstraint',lower_limit,upper_limit,ICT_channel)

class PlasmaStateConstraint(Constraint):
    def __init__(self,plasma_on=True):
        Constraint.__init__(self,plasma.match_plasma_state,'PlasmaStateConstraint',plasma_on)
    
        
