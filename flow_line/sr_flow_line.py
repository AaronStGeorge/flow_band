import sys

src_directory = '../../VarGlaS/'
sys.path.append(src_directory)
src_directory = '../data'
sys.path.append(src_directory)

from dolfin            import *
from sr_data           import SrData
from pylab             import array
from matplotlib        import pyplot
from scipy.interpolate import interp1d


# initialise modified utilities.DataInput object
sr = SrData()

# create bed slope functions for each component direction
grad_u = project(grad(sr.B), VectorFunctionSpace(sr.mesh, 'Lagrange', 1))
Q      = FunctionSpace(sr.mesh, 'Lagrange', 1)
u,v    = grad_u.split()
u      = project(u,Q)
v      = project(v,Q)

def rhs_func(t,y):
    dx = -u(y[0],y[1])
    dy = -v(y[0],y[1])
    dn = sqrt(dx**2 + dy**2)
    # slope cutoff after which bed is largly flat
    if dn>0.001:
      return array([dx/dn,dy/dn])
    else:
      return array([0.0,0.0])

class FlowLineSr:

  def __init__(self, y0, kx=3):
    """
    Creates a spline-interpolation expression for flow line  created from the
    inverse gradient of bed slope for the antarctic dry valleys study region.
    Optional arguments <kx> determine order of approximation (default cubic).
    """

    self.y0 = y0

    # Initial arclength
    t0 = 0.0

    # Intialize ode solver
    from scipy.integrate import ode
    r = ode(rhs_func).set_integrator('vode',method='adams')
    r.set_initial_value(y0,t0)
    dt = 100.
    # Edge to ege the dry valleys study region is about 3,000,000 meters
    t_end = 3000000. 
    
    # x,y positions
    self.pos_x = [y0[0]]
    self.pos_y = [y0[1]]
    
    # arclength
    rs = [0.0]
    
    # Query data structures
    Blist = [sr.B(y0[0],y0[1])]
    
    # Integrate through "time" (really arclength)
    while r.successful() and r.t<t_end and any(rhs_func(0.0,r.y)):
      r.integrate(r.t + dt)
      self.pos_x.append(r.y[0])
      self.pos_y.append(r.y[1])
      rs.append(r.t)
    
      # Query dolfin functions and add to list
      Blist.append(sr.B(r.y[0],r.y[1]))
    
    # Convert lists into arrays 
    self.s = array(rs)
    self.B = array(Blist)

    # make extents available in namespace
    self.minimum = 0
    self.maximum = max(self.s)

    # Create spline
    self.spline = interp1d(self.s, self.B, kind=kx)

  def plot_fl(self):

    fig = pyplot.figure(figsize=(14,7))

    # Subplot 1
    ax1 = fig.add_subplot(121)
    ax1.set_title("Dry valleys elevation")
    ax1.imshow(sr.data['b'], 
                  extent=[sr.x_min, sr.x_max, sr.y_min, sr.y_max], 
                  origin='lower')
    fig.gca().add_artist(pyplot.Circle(self.y0, 100, color='r'))
    ax1.set_xlim(sr.x_min, sr.x_max)
    ax1.set_ylim(sr.y_min, sr.y_max)
    ax1.plot(self.pos_x, self.pos_y, color='r')
    
    # Subplot 2
    ax2 = fig.add_subplot(122)
    ax2.set_title("Bed elevation")
    ax2.set_xlabel("length (m)")
    ax2.set_ylabel("height (m)")
    ax2.yaxis.tick_right()
    ax2.yaxis.set_label_position('right')
    ax2.plot(self.s,self.B)

    pyplot.show()

  def spline_expression_sr_bed(self,element):
    
    # Expression object for use with FEniCS
    class SrFlowBand(Expression):
    
      def eval(s, values, x):
        values[0] = self.spline(x[0]) / 5.

      def value_shape(self):
        return (1,)
    
    return SrFlowBand(element=element)
