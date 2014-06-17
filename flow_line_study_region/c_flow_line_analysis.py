import sys
import os
src_directory = '../../VarGlaS/'
sys.path.append(src_directory)

from pylab   import *
from dolfin  import *
from plot    import plotIce
from data.data_factory    import DataFactory
from src.utilities        import DataInput
import pylab
from sr_data import mesh, bed


Q = FunctionSpace(mesh,"CG",1)
V = MixedFunctionSpace([Q]*4)

# Import velocity data
U = Function(V)
File("assimilation_results/U_"+data_set+".xml") >> U
# Components:
u,u2,v,v2 = split(U)
u= project(u,Q)
v= project(v,Q)

u_mag = project(sqrt(u**2 + v**2))

# From the data files
H = Function(Q,data_dir + 'H.xml')
S = Function(Q,data_dir + 'S.xml')
B = Function(Q,data_dir + 'Bed.xml')

# derivative of position
def rhs_func(t,y):
    dx = -u(y[0],y[1])
    dy = -v(y[0],y[1])
    dn = pylab.sqrt(dx**2 + dy**2)
    # Velocity cutoff for having reached the divide.
    if dn>5.:
        return array([dx/dn,dy/dn])
    else:
        return array([0.0,0.0])
t0 = 0.0

#Jakobshavn
y0 = [-425597 + -1000.0,-2248477 + 1000.0]

from scipy.integrate import ode
r = ode(rhs_func).set_integrator('vode',method='adams')
r.set_initial_value(y0,t0)
dt = 1000.0
t_end = 10000000.

# x,y positions
pos_x = [y0[0]]
pos_y = [y0[1]]

# ARCLENGTH
rs = [0.0]

# Query data structures
Slist = [S(y0[0],y0[1])]
Blist = [B(y0[0],y0[1])]
Ulist = [-pylab.sqrt(u(y0[0],y0[1])**2 + v(y0[0],y0[1])**2)]

# Integrate through "time" (really arclength)
while r.successful() and r.t<t_end and any(rhs_func(0.0,r.y)):
    r.integrate(r.t + dt)
    pos_x.append(r.y[0])
    pos_y.append(r.y[1])
    rs.append(r.t)

# Query dolfin functions and add to list
    Slist.append(S(r.y[0],r.y[1]))
    Blist.append(B(r.y[0],r.y[1]))
    Ulist.append(-pylab.sqrt(u(r.y[0],r.y[1])**2 + v(r.y[0],r.y[1])**2))

# turn lists into arrays 
X = array(pos_x)
Y = array(pos_y)
s = array(rs)
S = array(Slist)
B = array(Blist)
U = array(Ulist)
fl    = array([pos_x, pos_y])

plotIce(u_mag, 'jet', dbm, fl, scale='log', name='U', units='m/a',
        numLvls=12, tp=False, tpAlpha=0.5)

pylab.plot(s,B,'k',s,S,'k')
pylab.show()
