# 
# .. _demo_poisson_equation:
# 
# Poisson equation
# ================

from dolfin import *

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


print("element degree:", degree_custom)
print("# of refinements:", n_refine)

print()


file_error = open('data_error_py.txt', 'a')
file_error.write("n_refine n_dofs error_l2 error_h1_semi error_h2_semi cpu_time\n")

i = 0
k = 1

n_cells = 1

grid_size = 1.0

n_dofs = 1


array_error = np.zeros((n_refine,3))
array_order_of_convergence = np.zeros((n_refine,3))
array_time_cpu = np.zeros((n_refine,1))


while i<=n_refine-1:
    
    t = time.time()
    
    print("########################################")
    print(i+1,"th refinement")
    print("    # of rectangular cells in one direction:",k)
    
    # Create mesh and define function space
    mesh = UnitSquareMesh(k, k,"crossed")
    #mesh = RectangleMesh(Point(0.0,0.0), Point(1.0,1.0), k, k, "crossed")

    gdim = mesh.geometry().dim()

    n_cells = mesh.num_cells()
    
    grid_size = 1.0/(2.0*k)
    
    coor = mesh.coordinates()
    
    print("    coordinates of the grid")
    for index_coor in range(len(coor)):
        print("        {:2.2e}".format(coor[index_coor][0]))
    
    print("    dimension of the geometry:", gdim)
    print("    # of active cells:", n_cells)
    #print("# of edges:", mesh.num_edges())
    #print("coordinates of the mesh\n",mesh.coordinates())

    print("    grid size of the active cell:", grid_size)
    

    V = FunctionSpace(mesh, "Lagrange", degree_custom)

    #print(V.dolfin_element())
    
    dofmap = V.dofmap()
    dofs = dofmap.dofs()
    n_dofs = len(dofs)
    
    print("    # of dofs:",n_dofs)
    #print(dofs)
    
    ## Get coordinates as len(dofs) x gdim array
    dofs_x = V.tabulate_dof_coordinates().reshape((-1, gdim))

    print("    coordinates of dofs(", n_dofs, "):")
    for dof, dof_x in zip(dofs, dofs_x):
        print ("        ", dof, ':', dof_x)

    # Define Dirichlet boundary (x = 0 or x = 1)
    def boundary(x):
        #print("boundary()", end = " ")
        #print(x)
        return x[0] < DOLFIN_EPS or x[0] > 1.0 - DOLFIN_EPS

    #print('DOLFIN_EPS:', DOLFIN_EPS)


    #print("Dirichlet boundary")
    #for x in mesh.coordinates():
        #if(boundary(x)):
            #print(x)
    

    # Define boundary condition
    #u0 = Constant(0.0)
    #u0 = Expression(("1.0 + pow(x[0]-0.5,2) + pow(x[1]-0.5,2)"), degree = 2)
    u0 = Expression(("exp(-pow(x[0]-0.5,2))"), degree = degree_custom)


    bc = DirichletBC(V, u0, boundary)


    # Define variational problem
    u = TrialFunction(V)
    v = TestFunction(V)
    #f = Expression("10*exp(-(pow(x[0] - 0.5, 2) + pow(x[1] - 0.5, 2)) / 0.02)", degree=2)
    #f = Expression("-4", degree=2)
    f = Expression("-exp(-pow(x[0]-0.5,2))*(pow(2*(x[0]-0.5),2)-2)", degree=degree_custom)



    #g = Expression("sin(5*x[0])", degree=2)
    #g = Expression("1.0", degree=2)                         # already taking the normal vector into account
    g = Expression("0.0", degree=degree_custom)


    a = inner(grad(u), grad(v))*dx
    L = f*v*dx + g*v*ds

    # Compute solution
    u = Function(V)
    solve(a == L, u, bc)
    
    u_P1 = project(u, V)
    u_nodal_values = u_P1.vector()

    print("nodal values of u")
    for index_nodal in range(len(u_nodal_values)):
        print("    ({:2.2e}, {:2.2e}) {:2.2e}".format(dofs_x[index_nodal][0], dofs_x[index_nodal][1], u_nodal_values[index_nodal]))


    # Save solution in VTK format
    file = File("poisson.pvd")
    file << u
    
    # Compute errors
    print("Computing the error")
    
    x = SpatialCoordinate(mesh)
    uexact = exp(-pow(x[0]-0.5,2))
    M = (u - uexact)**2*dx(degree=5)                            # adopted in undocumented/poisson-disc.py
    M0 = uexact**2*dx(degree=5)
    l2_error_u = sqrt(assemble(M))
    l2_norm_u = sqrt(assemble(M0))
    
    grad_diff_u = grad(u-uexact)
    h1_semi_error_u = sqrt(assemble(inner(grad_diff_u,grad_diff_u)*dx))
    
    grad_grad_diff_u = grad(grad(u-uexact))
    h2_semi_error_u = sqrt(assemble(inner(grad_grad_diff_u,grad_grad_diff_u)*dx))
    
    grad_u = grad(u)
    h1_seminorm_u = sqrt(assemble(inner(grad_u,grad_u)*dx))
    
    grad_grad_u = grad(grad(u))
    h2_seminorm_u = sqrt(assemble(inner(grad_grad_u,grad_grad_u)*dx))
        
    print("    Using assemble()")
    print("        l2_error_u:", "{:2.2e}".format(l2_error_u))
    print("        h1_semi_error_u:", "{:2.2e}".format(h1_semi_error_u))
    print("        h2_semi_error_u:", "{:2.2e}".format(h2_semi_error_u))
    
    print()
    print("        l2_norm_u:", "{:2.2e}".format(l2_norm_u))
    print("        h1_seminorm_u:","{:2.2e}".format(h1_seminorm_u))     
    print("        h2_seminorm_u:","{:2.2e}".format(h2_seminorm_u))     
    
    
    
    print("    Using norm()")
    print("        norm(u):","{:2.2e}".format(norm(u)))
    print("        norm(u,'H10'):","{:2.2e}".format(norm(u,'H10')))
    #print("    norm(u,'Hdiv0'):","{:2.2e}".format(norm(u,'Hdiv0')))
    
    
    
    array_error[i][0]=l2_error_u
    array_error[i][1]=h1_semi_error_u
    array_error[i][2]=h2_semi_error_u
    
    if i>0 and i<n_refine:
        for k in range(3):
            array_order_of_convergence[i][k] = math.log(array_error[i-1][k]/array_error[i][k],10)/math.log(2,10)

    time_cpu_elapsed = time.time() - t
    
    array_time_cpu[i]=time_cpu_elapsed
    
    print("time elapsed:", "{:6.4f}".format(time_cpu_elapsed))
    
    file_error.write("%s %s %0.2e %0.2e %0.2e %0.2e\n" %(i+1, n_dofs, l2_error_u, h1_semi_error_u, h2_semi_error_u, time_cpu_elapsed))
    
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


# Plot solution
#import matplotlib.pyplot as plt
#plot(u)
#plt.show()
