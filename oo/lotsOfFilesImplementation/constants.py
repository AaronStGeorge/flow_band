from dolfin import *

# Model parameters
curve    = 1
shelf    = True
floating = True


# Using the notation in the paper (not the book)
n = 3.0       # Flow law exponent
g = 9.8       # Gravitational constant
rho   = 0.917 # Density of ice
rho_w = 1.025 # Density of sea water

mu  = 1.0     # "a variable friction parameter"
A_s = 0.01    # "a sliding constant"  (B_s in the book)
p   = 1.0     # Exponent on effective pressure term (q in the book)
B_s = 540.0   # Flow law constant (B in the book)

N    = 100  # Number of finite elements


# Parameters specific to the two curves in "Beyond back stress: ..."
if curve == 1:
  M = Constant(0.3)      # Accumulation
  if shelf:
    maximum_L = 498.00e3 # Final length
    c =  78.46           # Surface elevation at the calving front
  else:
    maximum_L = 491.36e3 # Final length (distance to the grounding line)
    c = 121.56           # Surface elevation at the terminus
else:
  M = Expression("5.3/(1 + 200*exp(0.05e-3*(x[0]-500e3))) - 5") # Accumulation
  maximum_L = 1e10     # Maximum length
  if shelf:
    c = 31.67          # Surface elevation at the calving front
  else:
    c = 115.90         # Surface elevation at the terminus
