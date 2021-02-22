# 
# .. _demo_mixed_poisson:
# 
# Mixed formulation for Poisson equation
# ======================================

from dolfin import *
import matplotlib.pyplot as plt

import getopt, sys
import math
import time

import numpy as np


full_cmd_arguments = sys.argv

argument_list = full_cmd_arguments[1:]


print('Number of arguments:', len(argument_list), 'arguments.')
print('Argument List:', str(argument_list))

print("")

degree_custom = int(argument_list[0])
n_refine = int(argument_list[1])


file_error = open('data_error_fenics.txt', 'a')
file_error.write("n_refine n_dofs_bdm n_dofs_dg error_l2 error_h1_semi error_h2_semi cpu_time\n")

level_initial_refinement=1
i = 0
k = pow(2,level_initial_refinement)

n_cells = 1

grid_size = 1.0

n_dofs_bdm = 1
n_dofs_dg = 1

array_error = np.zeros((n_refine+1,3))
array_order_of_convergence = np.zeros((n_refine+1,3))
array_time_cpu = np.zeros((n_refine+1,1))

l2_error_u = 1.0
l2_error_sigma = 1.0
l2_error_grad_sigma = 1.0

print("Initial refinement level:", level_initial_refinement)
print("element degree:", degree_custom)
print("# of refinements:", n_refine)

print()


while i<=n_refine:
        
    t = time.time()
    
    print("########################################")
    print(i,"th refinement")                                                     # We start with zeroth refinement
    
    print("    # of rectangular cells in one direction:",k)
    # Create mesh and define function space
    mesh = UnitSquareMesh(k, k,"crossed")
    #mesh = RectangleMesh(Point(0.0,0.0), Point(1.0,1.0), k, k, "crossed")

    gdim = mesh.geometry().dim()
    n_cells = mesh.num_cells()

    coor = mesh.coordinates()

    #print("    coordinates of the nodes of the grid")
    #for index_coor in range(len(coor)):
        #print("        {:2.2e} {:2.2e}".format(coor[index_coor][0],coor[index_coor][1]))

    print("    dimension of the geometry:", gdim)
    print("    # of active cells:", n_cells)
        
    # Define finite elements spaces and build mixed space
    BDM = FiniteElement("BDM", mesh.ufl_cell(), degree_custom)
    DG  = FiniteElement("DG", mesh.ufl_cell(), degree_custom-1)
    W = FunctionSpace(mesh, BDM * DG)

    Space_BDM = FunctionSpace(mesh, "BDM", degree_custom)
    Space_DG = FunctionSpace(mesh, "DG", degree_custom-1)
    Space_CG = VectorFunctionSpace(mesh, "CG", degree_custom)

    dofmap_bdm = Space_BDM.dofmap()
    dofs_bdm = dofmap_bdm.dofs()
    n_dofs_bdm = len(dofs_bdm)

    dofmap_dg = Space_DG.dofmap()
    dofs_dg = dofmap_dg.dofs()
    n_dofs_dg = len(dofs_dg)

    print("    # of dofs of the velocity:",n_dofs_bdm)
    print("    # of dofs of the pressure:",n_dofs_dg)
    #print(dofs)

    ## Get coordinates as len(dofs) x gdim array
    #dofs_x_bdm = Space_BDM.tabulate_dof_coordinates().reshape((-1, gdim))
    dofs_x_dg = Space_DG.tabulate_dof_coordinates().reshape((-1, gdim))


    #print("    coordinates of dofs(", n_dofs_dg, "):")
    #for dof, dof_x in zip(dofs_dg, dofs_x_dg):
        #print ("        ", dof, ':', dof_x)


    # Define trial and test functions
    (sigma, u) = TrialFunctions(W)
    (tau, v) = TestFunctions(W)


    # Define source function
    #f = Expression("10*exp(-(pow(x[0] - 0.5, 2) + pow(x[1] - 0.5, 2)) / 0.02)", degree=2)
    f = Expression("-4", degree=degree_custom)
    #f = Expression("-exp(-pow(x[0]-0.5,2))*(pow(2*(x[0]-0.5),2)-2)", degree=degree_custom)

    #u0 = Constant(0.0)
    u0 = Expression("pow(x[0]-0.5,2) + pow(x[1]-0.5,2)", degree=degree_custom)
    #u0 = Expression(("exp(-pow(x[0]-0.5,2))"), degree = degree_custom)

    n = FacetNormal(mesh)


    # Define variational form
    a = (dot(sigma, tau) + div(tau)*u + div(sigma)*v)*dx
    L = - f*v*dx + u0*dot(tau, n)*ds


    # Define function G such that G \cdot n = g
    class BoundarySource(UserExpression):
        def __init__(self, mesh, **kwargs):
            
            print("Initializing BoundarySource")
            
            self.mesh = mesh
            super().__init__(**kwargs)
        def eval_cell(self, values, x, ufc_cell):
            cell = Cell(self.mesh, ufc_cell.index)
            
            #print("cell name", end = ": ")
            #print(cell.cellname())
            
            n = cell.normal(ufc_cell.local_facet)
            
            #print("n", end = ": ")
            #print(n)
            
            #g = sin(5*x[0])
            g = 1.0
            #g = 0.0
            values[0] = g*n[0]
            values[1] = g*n[1]
            
            #print("values", end = ": ")
            #print(values)
            
        def value_shape(self):
            return (2,)


    G = BoundarySource(mesh, degree=degree_custom)


    # Define essential boundary
    def boundary(x):                                                        # note the essential BC is the Neumann BC
        return x[1] < DOLFIN_EPS  or x[1] > 1.0 - DOLFIN_EPS


    bc = DirichletBC(W.sub(0), G, boundary)

    # Compute solution
    w = Function(W)
    solve(a == L, w, bc)
    (sigma, u) = w.split()

    sigma_projected = project(sigma, Space_BDM)
    values_sigma_projected = sigma_projected.vector()

    u_projected = project(u, Space_DG)
    values_u_projected = u_projected.vector()


    #print("values of sigma on dofs,", len(values_sigma_projected), "in total")
    #for index_nodal in range(len(values_sigma_projected)):
        #print("    {:2.2e}".format(values_sigma_projected[index_nodal]))


    #print("values of u on dofs,", len(values_u_projected), "in total")
    #for index_nodal in range(len(values_u_projected)):
        #print("    ({:2.2e}, {:2.2e}) {:2.2e}".format(dofs_x_dg[index_nodal][0], dofs_x_dg[index_nodal][1], values_u_projected[index_nodal]))



    #sigma_interpolated_cg = interpolate(sigma, Space_CG)
    #values_sigma_interpolated_cg = sigma_interpolated_cg.vector()

    #print("nodal values of sigma", len(values_sigma_interpolated_cg))
    #for index_nodal in range(len(values_sigma_interpolated_cg)):
        #print("    {:2.2e}".format(values_sigma_interpolated_cg[index_nodal]))


    # Compute errors
    print("Computing the error")

    x = SpatialCoordinate(mesh)
    uexact = pow(x[0]-0.5,2) + pow(x[1]-0.5,2)
    #uexact = exp(-pow(x[0]-0.5,2))
    
    
    M = (u - uexact)**2*dx(degree=5)                            # adopted in undocumented/poisson-disc.py
    l2_error_u = sqrt(assemble(M))
    
    sigma_exact = as_vector([2.0*(x[0]-0.5),2.0*(x[1]-0.5)])
    #sigma_exact = as_vector([exp(-pow(x[0]-0.5,2))*(-2.0*(x[0]-0.5)),0.0])
    diff_sigma = sigma - sigma_exact
    l2_error_sigma = sqrt(assemble(inner(diff_sigma,diff_sigma)*dx))
    
    grad_diff_sigma = grad(diff_sigma)
    l2_error_grad_sigma = sqrt(assemble(inner(grad_diff_sigma, grad_diff_sigma)*dx))

    print("        l2_error_u:", "{:2.2e}".format(l2_error_u))
    print("        l2_error_sigma:", "{:2.2e}".format(l2_error_sigma))
    print("        l2_error_grad_sigma:", "{:2.2e}".format(l2_error_grad_sigma))

    array_error[i][0]=l2_error_u
    array_error[i][1]=l2_error_sigma
    array_error[i][2]=l2_error_grad_sigma
    
    if i>0 and i<=n_refine:
        for k in range(3):
            array_order_of_convergence[i][k] = math.log(array_error[i-1][k]/array_error[i][k],10)/math.log(2,10)
    
    

    file = File("mixed_poisson_sigma.pvd")
    file << sigma

    file = File("mixed_poisson_u.pvd")
    file << u
    
    
    time_cpu_elapsed = time.time() - t
    
    array_time_cpu[i]=time_cpu_elapsed
    
    print("time elapsed:", "{:6.4f}".format(time_cpu_elapsed))
        
    file_error.write("%s %s %s %0.2e %0.2e %0.2e %0.2e\n" %(i, n_dofs_bdm, n_dofs_dg, l2_error_u, l2_error_sigma, l2_error_grad_sigma, time_cpu_elapsed))
            
    i += 1
    k = pow(2,i)    


print("")
print("summary")
np.set_printoptions(formatter={'float': '{: 0.2e}'.format})

print("error")
for line in array_error:
    print(line)

print("order of convergence")
np.set_printoptions(formatter={'float': '{: 0.2f}'.format})
for line in array_order_of_convergence:
    print(line)
    
print("cpu time")
np.set_printoptions(formatter={'float': '{: 0.2e}'.format})
for line in array_time_cpu:
    print(line)
    
print()
    
file_error.close() 

# Plot sigma and u
#plt.figure()
#plot(sigma)

#plt.figure()
#plot(u)

#plt.show()
