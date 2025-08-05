import os
import sys
import numpy as np
import vector
import torch

def get_me( event, matrix2py_path, paramcard_path ):
    sys.path.append( matrix2py_path )
    import matrix2py
    matrix2py.initialisemodel( paramcard_path )
    alphas = 0.13
    nhel = -1 # means sum over all helicity     
    me2 = matrix2py.get_value(event, alphas, nhel)
    sys.path.remove( matrix2py_path )
    del sys.modules['matrix2py']
    return me2