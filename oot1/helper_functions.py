from dolfin import *
from matplotlib import pyplot
from numpy import array, linspace, empty
import numpy as np

def initial_h(h0,hL,xi,L):  # Smoothly join a parabola and a line at xi.
  if xi <= 0:
    return Expression('m*x[0]+b', m=(hL-h0)/L, b=h0, cell=interval)
  A = (h0-hL)/(xi-2*L)/xi
  m = 2*A*xi
  b = hL-m*L
  return Expression('x[0] < xi ? A*x[0]*x[0]+C : m*x[0]+b',
                     xi=xi, A=A, C=h0, m=m, b=b, cell=interval)

class HelperFunctions:

  def __init__(self, mesh, V, h_b, W, maximum_L):

    self.mesh = mesh
    self.V    = V

    # number of nodal points in mesh
    N = self.mesh.coordinates().size

    # Create variables for plotting the topography
    self.full_mesh = IntervalMesh(100, 0, maximum_L)
    self.full_x    = self.full_mesh.coordinates()
    self.full_V    = FunctionSpace(self.full_mesh, 'Lagrange', 1)
    self.full_bed  = map(project(h_b, self.full_V), self.full_x)
    self.full_w    = array(map(project(W, self.full_V), self.full_x))/1000.0
    self.full_x    = self.full_x * 0.001 # convert from meters to kilometers

    mesh_coordinates = interpolate(Expression("x[0]"), self.V).vector().array()
    self.indicies = []
    for i in range(N):
      for j in range(N):
        if mesh_coordinates[i] == self.mesh.coordinates()[j,0]:
          self.indicies.append(j)
          break
    del(mesh_coordinates)
    
    if len(self.indicies) != N:
      print '\nERROR: Correspondence of indices of h.vector.array to \
          mesh.coordinates failed.'
      sys.exit()

  def update_mesh(self,L,maximum_L,mesh,dt,h_u_dhdx, h_old):

    N = mesh.coordinates().size

    new_L = min(L + float(dt)*h_u_dhdx(L)[1], maximum_L)
    h_ = array(map(h_u_dhdx, mesh.coordinates()))[:,0]
    new_mesh_coordinates = linspace(0, new_L, N)
    h_on_new_mesh = empty(N)

    for i in range(N):
      if new_mesh_coordinates[i] < L:
        h_on_new_mesh[self.indicies[i]] = h_u_dhdx(new_mesh_coordinates[i])[0]
      else:
        h_on_new_mesh[self.indicies[i]] = h_[-1]

    h_old.vector()[:] = h_on_new_mesh
    mesh.coordinates()[:,0] = new_mesh_coordinates

    if dolfin.__version__[:3] == '1.3':
      mesh.bounding_box_tree().build(mesh)
    if dolfin.__version__[:3] == '1.4':
      mesh.bounding_box_tree().build(mesh)
    else:
      mesh.intersection_operator().clear()
    
    return new_L

  def smooth_step_conditional(self, c):
    """
    diffuses conditional (or any other function on the function space self.V for
    that matter) to create a smoothed step function.

    :param c  : Expression - conditional to be smoothed
    """

    # set boundary condition to conditional
    class Boundary(SubDomain):  
      def inside(self, x, on_boundary):
        return on_boundary
                
    # boundary condition
    boundary = Boundary()
    bc       = DirichletBC(self.V, c, boundary)

    # set diffusivity parameter to average cell size
    cs = CellSize(self.mesh)
    cs = project(cs,self.V)
    m  = cs.vector()*cs.vector()
    D  = m.sum()/m.size()

    # solve heat equation with dt=1 (therefore neglected)
    u = TrialFunction(self.V)
    v = TestFunction(self.V)
    a = u*v*dx + D*inner(nabla_grad(u), nabla_grad(v))*dx
    L = c*v*dx
    u = Function(self.V)

    solve(a == L, u, bc)

    return u
  

  def mtpl_plot(self, h, H, u, L, h_b, driving_stress, basal_drag, 
                lateral_drag, longitudinal_stress, t_float, maximum_L):

    def plot_details():
      pyplot.xlim(0,maximum_L/1000)
      pyplot.grid(True)

    ### Figure 1 ###
    pyplot.ion()
    pyplot.figure(1, figsize=(6,6)).subplots_adjust(left=0.175, right=0.95)
    pyplot.clf()
    pyplot.subplot(311)

    surface  = project(h,self.V)
    bottom   = project(h-H,self.V)
    velocity = project(u,self.V)

    surface_ = map(surface, self.mesh.coordinates())+[bottom(L)]
    bottom_  = map(bottom, self.mesh.coordinates())
    x = 0.001*self.mesh.coordinates().flatten()
    # plot the glacier top surface
    pyplot.plot(list(x)+[x[-1]],surface_,'b',label="top") 
    # plot the glacier bottom surface
    pyplot.plot(x,bottom_,'r',label="bottom")
    # plot the basal topography
    pyplot.plot(self.full_x, self.full_bed,'g',label="bed")
    #pyplot.legend(loc=4,prop={'size':6})

    def water_plot(x):
        if x <= L:
            return surface(x) >= 0 or h_b(x) >= 0
        else:
            return h_b(x) >= 0

    water = map(lambda x: 0, self.full_x)
    mask  = map(water_plot, self.full_mesh.coordinates())
    water = np.ma.masked_array(water,mask)

    pyplot.plot(self.full_x, water, 'c') # plot the water surface

    #if 'f' in command_line_arguments: # plot the height at which ice should float
    #  full_float = map(project(-h_b*(rho_w/rho-1), 
    #                           self.full_V), 1000*self.full_x)
    #  pyplot.plot(self.full_x, full_float, '--k')

    pyplot.ylabel('Height above sea level (m)')
    pyplot.ylim(100,500)
    plot_details()

    print '\nElevation at the divide is %f, thickness is %f\n' % (surface(0), surface(0)-bottom(0))

    pyplot.subplot(312)
    pyplot.plot(x,map(velocity,self.mesh.coordinates()))
    pyplot.plot(self.full_x,  self.full_w, 'g')
    pyplot.plot(self.full_x, -self.full_w, 'g')
    pyplot.xlabel('Distance from ice divide (km)')
    pyplot.ylabel('Velocity (m/a)')
    #pyplot.ylim(-50,300)
    plot_details()


    pyplot.subplot(313)
    pyplot.plot(x,map(t_float,self.mesh.coordinates()))
    pyplot.ylim(-.5,1.5)
    plot_details()

    """
    pyplot.subplot(414)
    pyplot.plot(x,map(project(H_t,V),mesh.coordinates()))
    pyplot.plot(x,map(project(H,V),mesh.coordinates()))
    #pyplot.ylim(-.5,1.5)
    pyplot.ylim(-100,2000)
    plot_details()
    """

    ### Figure 2 ###

    pyplot.figure(2, figsize=(6,6))
    pyplot.clf()

    tau_d   = project(-driving_stress,self.V)
    tau_b   = project(basal_drag,self.V)
    tau_lat = project(lateral_drag,self.V)
    tau_lon = project(-longitudinal_stress.dx(0),self.V)

    ## Plot the surface mass balance
    #if 'M' in command_line_arguments:
    #  accumulation = project(M,self.V)
    #  pyplot.subplot(211)
    #  pyplot.plot(x,map(accumulation,self.mesh.coordinates()))
    #  pyplot.ylabel('Surface mass balance (m/yr)')
    #  plot_details()
    #  pyplot.subplot(212)
    #  pyplot.gcf().subplots_adjust(left=0.175, right=0.95)

    # Plot terms in the force balance
    pyplot.plot(x,map(tau_d,self.mesh.coordinates()),label=r'$\tau_d$')
    pyplot.plot(x,map(tau_b,self.mesh.coordinates()),label=r'$\tau_b$')
    pyplot.plot(x,map(tau_lat,self.mesh.coordinates()),label=r'$\tau_\perp$')
    pyplot.plot(x,map(tau_lon,self.mesh.coordinates()),label=r'$\tau_-$')
    pyplot.ylim(-10,200)
    pyplot.legend(loc='upper right').get_frame().set_alpha(0.5)
    pyplot.xlabel('Distance from ice divide (km)')
    plot_details()

    #### Terminate or pause if appropriate
    ###if 'i' in command_line_arguments: # Only the initial condition is plotted
    ###  break

    #print '\ntime = %f\n' % ((i+1)*float(dt))

    #if 'p' in command_line_arguments: # Pause
    #  response = raw_input('Press ENTER (or enter "q" to quit, "l" to loop) ')
    #  if response == 'q': sys.exit()
    #  if response == 'l': command_line_arguments.remove('p')
    #else:
    pyplot.draw()
