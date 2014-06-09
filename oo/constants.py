"""
This module contains the classes that hold known 
physical constants relevant to the simulations and a class to create new 
constants

Adopted from VarGlas modified for flow band model
"""

class PhysicalConstant(float):
  """
  This class allows the creation of new floating point physical constants.
      
  :param float value: Value of the physical constant
  :param description: Description of the physical constant
  :param units: Units of the physical constant
  """
  def __new__(cls, value = 0.0, description = None, units = None):
    """
    Creates a new PhysicalConstant object
    """
    ii = float.__new__(cls,value)
    ii.description = description
    ii.units = units
    return ii

class FlowBandParameters(object):
  """
  This class contains the default physical parameters used in modeling
  the ice sheet.
  
  :param params: Optional dictionary object of physical parameters
  """
  def __init__(self,params=None):
    if params:
      self.params = params
    else:
      self.params = self.get_default_parameters()
      
  def globalize_parameters(self, *namespace_list):
    """
    This function converts the parameter dictinary into global PhysicalContstant
    objects
    
    :param namespace: Optional namespace in which to place the global variables
    """
    
    for namespace in namespace_list:
      for param in self.params.iteritems():
        vars(namespace)[param[0]] = PhysicalConstant(param[1][0],
                                                     param[1][1],
                                                     param[1][2])

  def get_default_parameters(self):
    """
    Creates a dictionary of default physical constants and returns it
    
    :rtype: Python dictionary
    """
    d_params = \
    {'n'      : (3.0,     \
        'viscosity nonlinearity parameter (flow law exponent)','dimensionless'),
     'rho'      : (.917,    'ice density','g cm^{-3}'),
     'rho_w'    : (1.025,   'density sea water', 'cg cm^{-3}'),
     'g'        : (9.81,    'gravitational acceleration','m s^{-2}'),
     'mu'       : (1.0,     'a variable friction parameter', '??'),
     'A_s'      : (.01,     'a sliding constant (B_s in the book)', '??'),
     'p'        : (1.0,     \
             'Exponent on effective pressure term (q in the book)','??'),
     'B_s'      : (540.0,   'Flow law constant (B in the book)', '??'),
     'L'        : (100e3,   'Initial length', 'meters'),
     'maximum_L': (1e10,    'maximum length', 'meters'),
     'c'        : (115.90,  'surface elevation at the calving front', 'meters')
    }

    return d_params
