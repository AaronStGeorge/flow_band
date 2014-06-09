from constants import FlowBandParameters
import sys
import dolfin as d
from dolfin import *
import numpy as npy
from matplotlib import pyplot

class Model(object):
  """ 
  Instance of a 1D flow band ice model.
  """

  def __init__(self, h_b, W, M, floating=True):
    """
    Sets the geometry of the surface and bed of the ice sheet.
 
    :param h_b  : Expression representing the base of the mesh
    :param W    : Expression representing the half width
    :param M    : Expression representing surface mass balance
    """
    
    # initialize parameters 
    params   = FlowBandParameters()
    params.globalize_parameters(self) 
    self.dt  = Constant(250)
    self.N   = 100 # number of finite elements
    self.h_b = h_b
    self.W   = W
    self.floating = floating

    # create mesh
    self.mesh = IntervalMesh(self.N,0,self.L)

    # create function spaces
    order = 1  # Linear finite elements
    self.V  = FunctionSpace(self.mesh, 'Lagrange', order)
    self.V3 = VectorFunctionSpace(self.mesh, 'Lagrange', order, 3)

    # create correspondence list between V.vector.array and mesh.coordinates
    # used in plotting functions
    mesh_coordinates = interpolate(Expression("x[0]"), self.V).vector().array()
    self.indices = []
    for i in range(self.N+1):
      for j in range(self.N+1):
        if mesh_coordinates[i] == self.mesh.coordinates()[j,0]:
          self.indices.append(j)
          break
    del(mesh_coordinates)

    if len(self.indices) != self.N+1:
      print '\nERROR: Correspondence of indices of h.vector.array \
          to mesh.coordinates failed.'
      sys.exit()

    self.initialize_bc()
    self.initialize_eqs()
    self.initialize_plot()

  def initialize_bc(self):
    
    boundary = MeshFunction("size_t", self.mesh, self.mesh.topology().dim()-1)
    self.ds = Measure('ds')[boundary]
    
    class Divide(d.SubDomain):
      def inside(self, x, on_boundary):
        return on_boundary and d.near(x[0], 0)
    
    class Terminus(d.SubDomain):
      def inside(s, x, on_boundary):
        return on_boundary and d.near(x[0], self.L)

    INTERIOR, DIVIDE, self.TERMINUS = 99, 0, 1
    boundary.set_all(INTERIOR)
    Divide().mark(boundary, DIVIDE)
    Terminus().mark(boundary, self.TERMINUS)

    self.boundary_conditions = [
      #h(L)=c
      DirichletBC(self.V3.sub(0), Constant(self.c),boundary, self.TERMINUS),
      DirichletBC(self.V3.sub(1), d.Constant(0), boundary, DIVIDE),#u(0) = 0
      DirichletBC(self.V3.sub(2), d.Constant(0), boundary, DIVIDE)]#dh/dx(0) = 0

  def initialize_eqs(self):
    """
    """


    # initial height
    h = self.initial_h(120+0.0003*self.L, self.c, 0.4*self.L, self.L)

    self.h_old = d.project(h, self.V)
    # initial guess of velocity
    u = d.project(d.Expression('x[0]/L', L=1000*self.L), self.V)     
    self.h_u_dhdx = d.project(d.as_vector((h,u,h.dx(0))), self.V3)
    self.h,self.u,self.dhdx = d.split(self.h_u_dhdx)

    # initialize floating conditional
    class FloatBoolean(Expression):
      """
      creates boolean function over length of ice, true when ice is floating.
      """
      def eval(s,value,x):
        if self.h_u_dhdx(x[0])[0] <= -h_b(x[0])*(self.rho_w/self.rho-1):
          value[0] = 1
        else:
          value[0] = 0

    # floating conditional
    fc = self.smooth_step_conditional(FloatBoolean())

    # A useful expression for the effective pressure term
    reduction = fc*(self.rho_w/self.rho)*self.h_b
    
    # H (ice thickness) conditional on floating
    if self.floating:
      # archimedes principle, H (ice thickness)
      self.H = fc * (h/(1-self.rho/self.rho_w)) + (1-fc) * (self.h-self.h_b) 
    else:
      # ice thickness = surface elevation - bed elevation
      self.H = self.h - self.h_b  

    H = self.H
    h = self.h
    W = self.W
    u = self.u
    n = self.n

    # basal drag
    self.basal_drag = self.mu*self.A_s*(self.H+reduction)**self.p*u**(1/n)

    if self.floating:
      self.basal_drag = (1-fc) * self.basal_drag

    # define variational problem
    testFunction  = TestFunction(self.V3)
    trialFunction = TrialFunction(self.V3)
    phi1, phi2, phi3 = split(testFunction)

    self.driving_stress = self.rho*self.g*H*self.dhdx
    self.lateral_drag = (self.B_s*H/W)*((n+2)*u/(2*W))**(1/n)
    
    force_balance = (self.driving_stress+self.basal_drag+self.lateral_drag)*phi2 
    
    ####????####################################################################
    mass_conservation = ((h-self.h_old)/self.dt + (H*u*W).dx(0)/W - M)*phi1
    ####????####################################################################
    
    F = (mass_conservation + force_balance)*d.dx
    
    ###????#####################################################################
    invariant_squared = ((n+2.)/(n+1.)*u/W)**2+u.dx(0)**2
    self.longitudinal_stress = \
            2*self.B_s*H*invariant_squared**((1-n)/(2*n))*u.dx(0)
    
    F += self.longitudinal_stress * phi2.dx(0) * d.dx
    F -= self.longitudinal_stress * phi2 * self.ds(self.TERMINUS)
    F += (self.h.dx(0)-self.dhdx)*phi3*d.dx
    ###????#####################################################################


    J = derivative(F, self.h_u_dhdx, trialFunction)
    problem =\
    NonlinearVariationalProblem(F, self.h_u_dhdx, self.boundary_conditions, J)
    self.solver  = NonlinearVariationalSolver(problem)

  def smooth_step_conditional(self,c):
    """
    diffuses conditional (or any other function for that matter) to create a 
    smoothed step function.

    :param c  : Expression conditional to be smoothed
    """

    # set boundary condition to conditional
    class Boundary(SubDomain):  # define the Dirichlet boundary
      def inside(self, x, on_boundary):
        return on_boundary
                
    boundary = Boundary()
    bc = DirichletBC(self.V, c, boundary)

    # set diffusivity parameter to average cell size
    cs = CellSize(self.mesh)
    cs = project(cs,self.V)
    m = cs.vector()*cs.vector()
    D = m.sum()/m.size()

    # solve heat equation with dt=1 (therefore neglected)
    u = TrialFunction(self.V)
    v = TestFunction(self.V)
    a = u*v*dx + D*inner(nabla_grad(u), nabla_grad(v))*dx
    L = c*v*dx

    u = Function(self.V)
    solve(a == L, u, bc)

    return u


  def initialize_plot(self):
    # Create variables for plotting the topography
    self.full_mesh = d.IntervalMesh(100, 0, 500e3)
    self.full_x    = self.full_mesh.coordinates()
    self.full_V    = d.FunctionSpace(self.full_mesh, 'Lagrange', 1)
    self.full_bed  = map(d.project(self.h_b, self.full_V), self.full_x)
    self.full_W    = \
            npy.array(map(d.project(self.W, self.full_V), self.full_x))/1000.
    self.full_x    = self.full_x * 0.001 # convert from meters to kilometers

  def plot_details(self):
    pyplot.xlim(0,500)
    labels = '0, ,100, ,200, ,300, ,400, ,500'.split(',')
    pyplot.xticks(range(0,501,50), labels)
    pyplot.grid(True)

  def plot(self,i):
    """
    plots two figures
    """
    ### Figure 1 ###

    h = self.h
    V = self.V
    H = self.H
    u = self.u
    mesh = self.mesh
    L = self.L

    pyplot.ion()
    pyplot.figure(1, figsize=(6,6)).subplots_adjust(left=0.175, right=0.95)
    pyplot.clf()
    pyplot.subplot(211)

    surface  = d.project(h,V)
    bottom   = d.project(h-H,V)
    velocity = d.project(u,V)

    surface_ = map(surface, mesh.coordinates())+[bottom(L)]
    bottom_  = map(bottom, mesh.coordinates())
    x = 0.001*mesh.coordinates().flatten()
    pyplot.plot(list(x)+[x[-1]],surface_,'b') # plot the glacier top surface
    pyplot.plot(x,bottom_,'b')                # plot the glacier bottom surface
    pyplot.plot(self.full_x, self.full_bed, 'g') # plot the basal topography
    pyplot.plot((max(L,self.x0)/1000,500), (0,0), 'c') # plot the water surface

    pyplot.ylabel('Height above sea level (m)')
    pyplot.ylim(-1000,3100)
    self.plot_details()

    print '\nElevation at the divide is %f, thickness is %f\n' % (surface(0),\
            surface(0)-bottom(0))

    pyplot.subplot(212)
    pyplot.plot(x,map(velocity,mesh.coordinates()))
    pyplot.plot(self.full_x,  self.full_W, 'g')
    pyplot.plot(self.full_x, -self.full_W, 'g')
    pyplot.xlabel('Distance from ice divide (km)')
    pyplot.ylabel('Velocity (m/a)')
    pyplot.ylim(-50,300)
    self.plot_details()

    ### Figure 2 ###

    pyplot.figure(2, figsize=(6,6))
    pyplot.clf()

    tau_d   = d.project(-self.driving_stress,V)
    tau_b   = d.project(self.basal_drag,V)
    tau_lat = d.project(self.lateral_drag,V)
    tau_lon = d.project(-self.longitudinal_stress.dx(0),V)

    # Plot terms in the force balance

    pyplot.plot(x,map(tau_d,mesh.coordinates()),label=r'$\tau_d$')
    pyplot.plot(x,map(tau_b,mesh.coordinates()),label=r'$\tau_b$')
    pyplot.plot(x,map(tau_lat,mesh.coordinates()),label=r'$\tau_\perp$')
    pyplot.plot(x,map(tau_lon,mesh.coordinates()),label=r'$\tau_-$')
    pyplot.ylim(-10,200)
    pyplot.legend(loc='upper right').get_frame().set_alpha(0.5)
    pyplot.xlabel('Distance from ice divide (km)')
    self.plot_details()

    ### Figure 3 ###  Plot the surface slope (only if requested)


    # Terminate or pause if appropriate

    print '\ntime = %f\n' % ((i+1)*float(self.dt))

    pyplot.draw()

  def update_mesh(self):
    """
    return:
        L : float new length
    side effect:
        updates h_old the projection of height onto the function space
    """
    # new L = oldL + dt*u(L) velocity in x direction at L
    new_L = min(self.L + float(self.dt)*self.h_u_dhdx(self.L)[1],self.maximum_L)
    h_ = npy.array(map(self.h_u_dhdx, self.mesh.coordinates()))[:,0]
    new_mesh_coordinates = npy.linspace(0, new_L, self.N+1)
    h_on_new_mesh = npy.empty(self.N+1)
  
    # if new mesh coordinate < L assign corresponding value from previous h
    # from h_u_dhdx else assign h at L
    for i in range(self.N+1):
      if new_mesh_coordinates[i] < self.L:
        h_on_new_mesh[self.indices[i]] = \
                self.h_u_dhdx(new_mesh_coordinates[i])[0]
      else:
        h_on_new_mesh[self.indices[i]] = h_[-1]
  
    # update mesh coordinates and h_old
    self.h_old.vector()[:] = h_on_new_mesh
    self.mesh.coordinates()[:,0] = new_mesh_coordinates
  
    # no idea what this shit does
    if d.dolfin.__version__[:3] == '1.3':
      self.mesh.bounding_box_tree().build(self.mesh)
    else:
      self.mesh.intersection_operator().clear()
  
    self.L = new_L
 

  def initial_h(self,h0,hL,xi,L):      
    """
    Smoothly join a parabola and a line at xi.

    :param h0 : float height of ice at 0 position
    :param hL : float height of ice at L (length of ice)
    :param xi : float 
    :param L  : float length of ice
    """
    if xi <= 0:
      return d.Expression('m*x[0]+b', m=(hL-h0)/L, b=h0, cell=d.interval)
    A = (h0-hL)/(xi-2*L)/xi
    m = 2*A*xi
    b = hL-m*L
    return d.Expression('x[0] < xi ? A*x[0]*x[0]+C : m*x[0]+b', 
                       xi=xi, A=A, C=h0, m=m, b=b, cell=d.interval)

  def solve(self, flag):
    """
    :param flag : "p" to plot
    """
    
    nSteps = 20 

    for i in range(nSteps):
      try:
        self.solver.solve()
      except RuntimeError as message:
        print message
        d.end()
        response = raw_input('Press ENTER to continue ("q" to quit) ')
        if response == 'q': sys.exit()
      
      self.plot(i)
      self.update_mesh()


if __name__ == "__main__":

  # Bed elevation
  bed = "1000/(1 + 200*exp(0.10e-3*(x[0]-250e3))) - 950"
  h_b = d.Expression(bed, cell=d.interval)  


  # half width of glacier
  W = d.Expression(\
          "25000/(1 + 200*exp(0.05e-3*(x[0]-200e3))) + 5000",cell=d.interval)
  # Accumulation (curve 2 from "Beyond Back Stress")
  M = d.Expression("5.3/(1 + 200*exp(0.05e-3*(x[0]-500e3))) - 5")   

  m = Model(h_b,W,M)

  m.solve(9)
