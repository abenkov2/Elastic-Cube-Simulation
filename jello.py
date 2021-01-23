import taichi as ti
import numpy as np

ti.init(arch=ti.gpu) # Try to run on GPU

quality = 1 # Use a larger value for higher-res simulations
n_particles, n_grid = 23680 * quality ** 2, 128 * quality
dx, inv_dx = 1 / n_grid, float(n_grid)
dt = 1e-5 / quality
p_vol, p_rho = (dx * 0.5)**2, 1
p_mass = p_vol * p_rho
E, nu = 2.7e3, 0.2 # Young's modulus and Poisson's ratio
mu_0, lambda_0 = E / (2 * (1 + nu)), E * nu / ((1+nu) * (1 - 2 * nu)) # Lame parameters

x = ti.Vector.field(3, dtype=float, shape=n_particles) # position
v = ti.Vector.field(3, dtype=float, shape=n_particles) # velocity
C = ti.Matrix.field(3, 3, dtype=float, shape=n_particles) # affine velocity field
F = ti.Matrix.field(3, 3, dtype=float, shape=n_particles) # deformation gradient
F_p = ti.Matrix.field(3, 3, dtype=float, shape=n_particles) # plastic deformation gradient
material = ti.field(dtype=int, shape=n_particles) # material id
Jp = ti.field(dtype=float, shape=n_particles) # plastic deformation
grid_v = ti.Vector.field(3, dtype=float, shape=(n_grid, n_grid, n_grid)) # grid node momentum/velocity
grid_m = ti.field(dtype=float, shape=(n_grid, n_grid, n_grid)) # grid node mass
gravity = ti.Vector.field(3, dtype=float, shape=())
attractor_strength = ti.field(dtype=float, shape=())
attractor_pos = ti.Vector.field(2, dtype=float, shape=())

@ti.func 
def kirchoff_FCR(F, R, J, mu, la):
  return 2 * mu * (F - R) @ F.transpose() + ti.Matrix.identity(float, 3) * la * J * (J - 1) #compute kirchoff stress for FCR model (remember tau = P F^T)
  
@ti.func
def clamp(x):
    return max(0.999, min(x, 1.00015))

@ti.kernel
def substep():
  
  #re-initialize grid quantities
  for i, j, k in grid_m:
    grid_v[i, j, k] = [0, 0, 0]
    grid_m[i, j, k] = 0
  
  # Particle state update and scatter to grid (P2G)
  for p in x: 
    
    #for particle p, compute base index
    base = (x[p] * inv_dx - 0.5).cast(int)
    fx = x[p] * inv_dx - base.cast(float)
    
    # Quadratic kernels  [http://mpm.graphics   Eqn. 123, with x=fx, fx-1,fx-2]
    w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]
    dw = [fx - 1.5, -2.0 * (fx - 1), fx - 0.5]

    
    # J_pl = (F_p[p]).determinant()
    mu, la = mu_0, lambda_0 #mu_0 * ti.exp(10 * (1 - J_pl)), lambda_0 * ti.exp(10 * (1 - J_pl)) #opportunity here to modify these to model other materials

    U, sig, V = ti.svd(F[p])
    J = 1.0
    
    

    for d in ti.static(range(3)):
      
      new_sig = sig[d, d]
      Jp[p] *= sig[d, d] / new_sig
      sig[d, d] = new_sig
      J *= new_sig
    
    
    #Compute Kirchoff Stress
    kirchoff = kirchoff_FCR(F[p], U@V.transpose(), J, mu, la)

    #P2G for velocity and mass AND Force Update!
    for i, j, k in ti.static(ti.ndrange(3, 3, 3)): # Loop over 3x3 grid node neighborhood
      offset = ti.Vector([i, j, k])
      dpos = (offset.cast(float) - fx) * dx
      weight = w[i][0] * w[j][1] * w[k][2]
      
      dweight = ti.Vector.zero(float,3)
      dweight[0] = inv_dx * dw[i][0] * w[j][1] * w[k][2]
      dweight[1] = inv_dx * w[i][0] * dw[j][1] * w[k][2]
      dweight[2] = inv_dx * w[i][0] * w[j][1] * dw[k][2]
 
      force = -p_vol * kirchoff @ dweight

      grid_v[base + offset] += p_mass * weight * (v[p] + C[p] @ dpos) #momentum transfer
      grid_m[base + offset] += weight * p_mass #mass transfer

      grid_v[base + offset] += dt * force #add force to update velocity, don't divide by mass bc this is actually updating MOMENTUM
  
  # Gravity and Boundary Collision
  for i, j, k in grid_m:
    if grid_m[i, j, k] > 0: # No need for epsilon here
      grid_v[i, j, k] = (1 / grid_m[i, j, k]) * grid_v[i, j, k] # Momentum to velocity
      
      grid_v[i, j, k] += dt * gravity[None] * 30 # gravity
      
      #wall collisions
      if i < 3 and grid_v[i, j, k][0] < 0:          grid_v[i, j, k][0] = 0 # Boundary conditions
      if i > n_grid - 3 and grid_v[i, j, k][0] > 0: grid_v[i, j, k][0] = 0
      if j < 3 and grid_v[i, j, k][1] < 0:          grid_v[i, j, k][1] = 0
      if j > n_grid - 3 and grid_v[i, j, k][1] > 0: grid_v[i, j, k][1] = 0
      if k < 3 and grid_v[i, j, k][2] < 0:          grid_v[i, j, k][2] = 0
      if k > n_grid - 3 and grid_v[i, j, k][2] > 0: grid_v[i, j, k][2] = 0
  
  # grid to particle (G2P)
  for p in x: 
    base = (x[p] * inv_dx - 0.5).cast(int)
    fx = x[p] * inv_dx - base.cast(float)
    w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1.0) ** 2, 0.5 * (fx - 0.5) ** 2]
    dw = [fx - 1.5, -2.0 * (fx - 1), fx - 0.5]
    new_v = ti.Vector.zero(float, 3)
    new_C = ti.Matrix.zero(float, 3, 3)
    new_F = ti.Matrix.zero(float, 3, 3)
    for i, j, k in ti.static(ti.ndrange(3, 3, 3)): # loop over 3x3 grid node neighborhood
      dpos = ti.Vector([i, j, k]).cast(float) - fx
      g_v = grid_v[base + ti.Vector([i, j, k])]
      weight = w[i][0] * w[j][1] * w[k][2]

      dweight = ti.Vector.zero(float,3)
      dweight[0] = inv_dx * dw[i][0] * w[j][1] * w[k][2]
      dweight[1] = inv_dx * w[i][0] * dw[j][1] * w[k][2]
      dweight[2] = inv_dx * w[i][0] * w[j][1] * dw[k][2]

      new_v += weight * g_v
      new_C += 4 * inv_dx * weight * g_v.outer_product(dpos)
      new_F += g_v.outer_product(dweight)
    v[p], C[p] = new_v, new_C
    x[p] += dt * v[p] # advection
    F[p] = (ti.Matrix.identity(float, 3) + (dt * new_F)) @ F[p] #updateF (explicitMPM way)
    

@ti.kernel
def reset():
  print("started reset")
  base = [60, n_grid - 12, 30]
  print("starting grid loop 1")
  cell = 0
  for z in range(1):
    for i in range(8):
      for j in range(8):
        for k in range(8):
          node = (base + ti.Vector([i, j, k])) * dx
          for g in range(cell, cell + 15):
            x[g] = node + [ti.random() * dx, ti.random() * dx, ti.random() * dx]
            v[g] = [0, 0, 0]
            F[g] = ti.Matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
            F_p[g] = ti.Matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
            Jp[g] = 1
            C[g] = ti.Matrix.zero(float, 3, 3)
          cell += 15
          print("cell: ", cell)
  base2 = [65, 35, 25]
  print("starting grid loop 2")
  for z in range(1):
    for i in range(10):
      for j in range(10):
        for k in range(10):
          #print('began grid loop')
          node = (base2 + ti.Vector([i, j, k])) * dx
          for g in range(cell, cell + 16):
            #print("g: ", g)
            x[g] = node + [ti.random() * dx, ti.random() * dx, ti.random() * dx]
            v[g] = [0, 0, 0]
            F[g] = ti.Matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
            F_p[g] = ti.Matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
            Jp[g] = 1
            C[g] = ti.Matrix.zero(float, 3, 3)
          cell += 16
          print("cell: ", cell)
  
  
    

  
print("[Hint] Use WSAD/arrow keys to control gravity. Use left/right mouse bottons to attract/repel. Press R to reset.")
reset()
gravity[None] = [0, -9.8, 0]

series_prefix = "proj3.ply"
for frame in range(120):
  for s in range(int(2e-3 // dt)):
    substep()
  np_pos = np.reshape(x.to_numpy(), (n_particles, 3))
  writer = ti.PLYWriter(num_vertices=n_particles)
  writer.add_vertex_pos(np_pos[:, 0], np_pos[:, 1], np_pos[:, 2])
  writer.export_frame_ascii(frame, series_prefix)
  print("Finished frame " + str(frame))
  

