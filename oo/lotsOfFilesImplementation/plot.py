from matplotlib import pyplot


class Plot:
  def __init__(self, h_b, W):

  # Create variables for plotting the topography
  self.full_mesh = IntervalMesh(100, 0, 500e3)
  self.full_x    = full_mesh.coordinates()
  self.full_V    = FunctionSpace(self.full_mesh, 'Lagrange', 1)
  self.full_bed  = map(project(h_b, self.full_V), self.full_x)
  self.full_W    = array(map(project(W, self.full_V), self.full_x))/1000.0
  self.full_x    = self.full_x * 0.001 # convert from meters to kilometers

  def plot_f(h,h_b,H,u,L):
    ########
    # Plot #
    ########
    
    ### Figure 1 ###
    pyplot.ion()
    pyplot.figure(1, figsize=(6,6)).subplots_adjust(left=0.175, right=0.95)
    pyplot.clf()
    pyplot.subplot(311)
    
    surface  = project(h,V)
    bottom   = project(h-H,V)
    velocity = project(u,V)
    
    surface_ = map(surface, mesh.coordinates())+[bottom(L)]
    bottom_  = map(bottom, mesh.coordinates())
    x = 0.001*mesh.coordinates().flatten()
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
    
    #pyplot.plot((max(L,x0)/1000,500), (0,0), 'c') # plot the water surface
    
    # plot the height at which ice should float
    if 'f' in command_line_arguments: 
      self.full_float = map(\
          project(-h_b*(rho_w/rho-1), self.full_V), 1000*self.full_x)
      pyplot.plot(self.full_x, self.full_float, '--k')
    
    pyplot.ylabel('Height above sea level (m)')
    pyplot.ylim(-1000,3100)
    plot_details()
    
    print '\nElevation at the divide is %f, thickness is %f\n'\
            % (surface(0), surface(0)-bottom(0))
    
    pyplot.subplot(312)
    pyplot.plot(x,map(velocity,mesh.coordinates()))
    pyplot.plot(self.full_x,  self.full_W, 'g')
    pyplot.plot(self.full_x, -self.full_W, 'g')
    pyplot.xlabel('Distance from ice divide (km)')
    pyplot.ylabel('Velocity (m/a)')
    pyplot.ylim(-50,300)
    plot_details()
    
    pyplot.subplot(313)
    pyplot.plot(x,map(t_float,mesh.coordinates()))
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
    
    tau_d   = project(-driving_stress,V)
    tau_b   = project(basal_drag,V)
    tau_lat = project(lateral_drag,V)
    tau_lon = project(-longitudinal_stress.dx(0),V)
    
    # Plot the surface mass balance
    
    if 'M' in command_line_arguments:
      accumulation = project(M,V)
      pyplot.subplot(211)
      pyplot.plot(x,map(accumulation,mesh.coordinates()))
      pyplot.ylabel('Surface mass balance (m/yr)')
      plot_details()
      pyplot.subplot(212)
      pyplot.gcf().subplots_adjust(left=0.175, right=0.95)
    
    # Plot terms in the force balance
    pyplot.plot(x,map(tau_d,mesh.coordinates()),label=r'$\tau_d$')
    pyplot.plot(x,map(tau_b,mesh.coordinates()),label=r'$\tau_b$')
    pyplot.plot(x,map(tau_lat,mesh.coordinates()),label=r'$\tau_\perp$')
    pyplot.plot(x,map(tau_lon,mesh.coordinates()),label=r'$\tau_-$')
    pyplot.ylim(-10,200)
    pyplot.legend(loc='upper right').get_frame().set_alpha(0.5)
    pyplot.xlabel('Distance from ice divide (km)')
    plot_details()
    
    # Plot the residual
    if 'r' in command_line_arguments:
      if 'o' in command_line_arguments:
        pyplot.plot(x,map(\
                lambda z: tau_b(z)+tau_lat(z), mesh.coordinates()), '--k')
        pyplot.plot(x,map(\
            lambda z: tau_d(z)-tau_b(z)-tau_lat(z), mesh.coordinates()), '--')
      else:
        pyplot.plot(x,map(\
        lambda z: tau_b(z)+tau_lat(z)+tau_lon(z), mesh.coordinates()), '--k')
        pyplot.plot(x,map(\
    da z: tau_d(z)-tau_b(z)-tau_lat(z)-tau_lon(z), mesh.coordinates()), '--')
    
    ### Figure 3 ###  Plot the surface slope (only if requested)
    
    if 's' in command_line_arguments:
      slope = project(dhdx,V)
      pyplot.figure(3, figsize=(6,6))
      pyplot.clf()
      pyplot.plot(x,map(slope,mesh.coordinates()),label='dh/dx')
      pyplot.grid(True)
      pyplot.legend(loc='best').get_frame().set_alpha(0.5)
    
    # Terminate or pause if appropriate
    
    if 'i' in command_line_arguments: # Only the initial condition is plotted
      break
    
    print '\ntime = %f\n' % ((i+1)*float(dt))
    
    if 'p' in command_line_arguments: # Pause
      response = raw_input('Press ENTER (or enter "q" to quit, "l" to loop) ')
      if response == 'q': sys.exit()
      if response == 'l': command_line_arguments.remove('p')
    else:
      pyplot.draw()
