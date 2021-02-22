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


if len(argument_list) != 7:
    raise Exception('Using <id_case> <segment> <id_coeff_diff> <segment> <degree> <initial refinement level> <total refinements>')


id_case = int(argument_list[0])

id_coeff_diff = int(argument_list[2])

degree_custom = int(argument_list[4])
level_initial_refinement=int(argument_list[5])
total_refinements = int(argument_list[6])


class Expression_U:
  def __init__(self, id_case):
    self.id_case = id_case    
  def func_value_u(self):
    switcher = {
        1: "pow(x[0]-0.5,2.0)",                                             #  + pow(x[1]-0.5,2)
        2: "exp(-pow(x[0]-0.5,2.0))"
    }
    return(switcher.get(id_case, "x[0]"))
  def func_gradient_u(self):
    switcher = {
        1: "2.0*(x[0]-0.5)",
        2: "exp(-pow(x[0]-0.5,2.0))*(-2.0*(x[0]-0.5))"
    }
    return(switcher.get(id_case, "x[0]"))
  def func_hessian_u(self):
    switcher = {
        1: "2.0",
        2: "exp(-pow(x[0]-0.5,2.0))*(pow(2*(x[0]-0.5),2.0)-2.0)"
    }
    return(switcher.get(id_case, "x[0]"))

        #2: "-exp(-pow(x[0]-0.5,2))*(pow(2*(x[0]-0.5),2)-2)*(1.0+x[0])-exp(-(pow(x[0]-0.5,2)))*(-2.0*(x[0]-0.5))"             
            # "exp(-(pow(x[0]-0.5, 2) + pow(x[1]-0.5, 2)))*(4.0*(pow(x[0]-0.5, 2) + pow(x[1]-0.5, 2) - 1.0))"


class Coeff_Diff:
  def __init__(self, id_coeff_diff):
    self.id_coeff_diff = id_coeff_diff
  def func_value_coeff_diff(self):
    switcher = {
        1: "1.0",
        2: "(1.0+x[0])",                # +x[0]*x[0]
        3: "exp(-pow(x[0]-0.5,2.0))",
        4: "(0.5 + pow(cos(x[0]), 2.0))"
    }
    return(switcher.get(id_coeff_diff, "x[0]"))      
  def func_gradient_coeff_diff(self):
    switcher = {
        1: "0.0",
        2: "(1.0)",                # +2.0*x[0]
        3: "exp(-pow(x[0]-0.5,2.0))*(-2.0*(x[0]-0.5))",
        4: "2.0*cos(x[0])*(-sin(x[0]))"
    }
    return(switcher.get(id_coeff_diff, "x[0]"))

class Expression_Normal_Derivative:
  def __init__(self, id_case, id_coeff_diff):
    self.id_case = id_case    
  def func_normal_derivative(self):                                   # not suitable for the varying normal vector !!!
    return(Coeff_Diff(id_coeff_diff).func_value_coeff_diff()+"*"+Expression_U(id_case).func_gradient_u())
        
  
    #switcher = {
        #1: "1.0",
        #2: "exp(-(pow(x[0]-0.5,2)))*(-2.0*(x[0]-0.5))"
    #}
    #return(switcher.get(argument, "x[0]"))



class Expression_F(Expression_U, Coeff_Diff):
  def __init__(self, id_case, id_coeff_diff):
    self.id_case = id_case
  def func_f(self):
    return("-1.0*(" + Coeff_Diff(id_coeff_diff).func_gradient_coeff_diff() + "*"+ Expression_U(id_case).func_gradient_u() + "+" + Coeff_Diff(id_coeff_diff).func_value_coeff_diff() + "*"+ Expression_U(id_case).func_hessian_u() + ")")
  
  
print("")
print('{:>25} {:>2}'.format("id_case:", id_case))
print('{:>25} {:>2}'.format("id_coeff_diff:", id_coeff_diff))
print('{:>25} {:>2}'.format("element degree:", degree_custom))
print('{:>25} {:>2}'.format("Initial refinement level:", level_initial_refinement))
print('{:>25} {:>2}'.format("# of total refinements:", total_refinements))
  
print("")
obj_expression_u = Expression_U(id_case)
print('{:>10} {:>2}'.format("u:", obj_expression_u.func_value_u()))
print('{:>10} {:>2}'.format(r"\nabla u:", obj_expression_u.func_gradient_u()))
print('{:>10} {:>2}'.format("\Delta u:", obj_expression_u.func_hessian_u()))

print("")
obj_coeff_diff = Coeff_Diff(id_coeff_diff)
print('{:>10} {:>2}'.format("D:", obj_coeff_diff.func_value_coeff_diff()))
print('{:>10} {:>2}'.format(r"\nabla D:", obj_coeff_diff.func_gradient_coeff_diff()))
    
print("")
obj_normal_derivative = Expression_Normal_Derivative(id_case, id_coeff_diff)        
print("normal_derivative:", obj_normal_derivative.func_normal_derivative())

print("")
obj_expression_f = Expression_F(id_case,id_coeff_diff)
print("f:", obj_expression_f.func_f())
 

print("")

file_error = open('data_error_fenics.txt', 'a')
file_error.write("current_refinement_level n_dofs error_l2 error_h1_semi error_h2_semi cpu_time\n")

current_refinement_level = level_initial_refinement
i = 0
n_rect_cell_one_direction = pow(2,level_initial_refinement)

n_cells = 1

grid_size = 1.0

n_dofs = 1


array_error = np.zeros((total_refinements,3))
array_order_of_convergence = np.zeros((total_refinements-1,3))
array_time_cpu = np.zeros((total_refinements,1))

l2_error_u = 1.0
h1_seminorm_u = 1.0
h2_seminorm_u = 1.0


while i<total_refinements:
    
    t = time.time()
    
    print("")
    print("########################################")
    print(i,"th refinement")                                # initial refinement level is denoted by i=0
    print("    # of rectangular cells in one direction:",n_rect_cell_one_direction)
    
    # Create mesh and define function space
    #mesh = UnitSquareMesh(n_rect_cell_one_direction, n_rect_cell_one_direction,"crossed")
    #mesh = RectangleMesh(Point(0.0,0.0), Point(1.0,1.0), n_rect_cell_one_direction, n_rect_cell_one_direction, "crossed")
    mesh = IntervalMesh(n_rect_cell_one_direction,0,1)

    gdim = mesh.geometry().dim()

    n_cells = mesh.num_cells()
    
    grid_size = 1.0/(2.0*n_rect_cell_one_direction)
    
    coor = mesh.coordinates()
    
    #print("    coordinates of the grid")
    #for index_coor in range(len(coor)):
        #print("        {:2.2e} {:2.2e}".format(coor[index_coor][0],coor[index_coor][1]))
    
    print("    dimension of the geometry:", gdim)
    print("    # of active cells:", n_cells)
    #print("# of edges:", mesh.num_edges())
    #print("coordinates of the mesh\n",mesh.coordinates())

    print("    grid size of the active cell:", grid_size)
    

    V = FunctionSpace(mesh, "Lagrange", degree_custom)
    
    
    dofmap = V.dofmap()
    dofs = dofmap.dofs()
    n_dofs = len(dofs)
    
    print("    # of dofs:",n_dofs)
    #print(dofs)
    
    ## Get coordinates as len(dofs) x gdim array
    dofs_x = V.tabulate_dof_coordinates().reshape((-1, gdim))

    #print("    coordinates of dofs(", n_dofs, "):")
    #for dof, dof_x in zip(dofs, dofs_x):
        #print ("        ", dof, ':', dof_x)
        
        
    #Quad = FunctionSpace(mesh, "CG", 2)                            # to do: nr. of quadrature points
        
    #xq = V.tabulate_all_coordinates(mesh).reshape((-1, gdim))
    #xq0 = xq[V.sub(0).dofmap().dofs()]
    
    
    

    # Define Dirichlet boundary (x = 0 or x = 1)
    def boundary_diri_location(x):
        #print("boundary()", end = " ")
        #print(x)
        return  x[0] < DOLFIN_EPS or x[0] > 1.0 - DOLFIN_EPS         # 

    #print('DOLFIN_EPS:', DOLFIN_EPS)


    print("Location of Dirichlet boundary")
    for x in mesh.coordinates():
        if(boundary_diri_location(x)):
            print(x)

    x = SpatialCoordinate(mesh)                             # important for eval()
    
    
    # Define boundary condition
    u0 = Expression(obj_expression_u.func_value_u(), degree = degree_custom)
                                                                            # change this for different problems, and also
                                                                            # 1. the rhs
                                                                            # 2. the projection of \nabla u on the boundary: \nabla u \cdot \underline(n), on line ~143
                                                                            # 3. the exact solution on line ~171
                                                                            
    bc_diri = DirichletBC(V, u0, boundary_diri_location)


    # Define variational problem
    u = TrialFunction(V)
    v = TestFunction(V)
    
    f = Expression(obj_expression_f.func_f(), degree=degree_custom)

    g = Expression(obj_normal_derivative.func_normal_derivative(), degree=degree_custom)


    #print(Cell(mesh, ufc_cell.index).normal(ufc_cell.local_facet))
    

    a = inner(eval(Coeff_Diff(id_coeff_diff).func_value_coeff_diff()) * grad(u), grad(v))*dx
    L = f*v*dx
    
    
    #print("LHS before applying BC:")
    #LHS= assemble(a)
    #print(LHS.array())
    
    #print("RHS before applying Neumann BC:")
    #RHS= assemble(L)
    #print(RHS.get_local())    
    
    
    n = FacetNormal(mesh)    
    
    L += g*dot(v,n[0])*ds
    
    #print("RHS after applying Neumann BC (no matter if it exists):")
    #RHS_after_neum_bc= assemble(L)
    #print(RHS_after_neum_bc.get_local())       
    
    
    #if x[1] > 1-DOLFIN_EPS:
        #L += g*v*ds
    
    # Compute solution
    u = Function(V)
    solve(a == L, u, bc_diri)
    
    
    #print("LHS after solving:")
    #LHS_after_solving= assemble(a)
    #print(LHS_after_solving.array())
    
    #print("RHS after solving:")
    #RHS_after_solving= assemble(L)
    #print(RHS_after_solving.get_local())
        
    
    
    #u_P1 = project(u, V)                        # since u is a function defined on V, we assume u_P1 is the exact representation of V
    #u_nodal_values = u_P1.vector()

    #print("nodal values of u")
    #for index_nodal in range(len(u_nodal_values)):
        #print("    ({:2.2e}, {:2.2e}) {:2.2e}".format(dofs_x[index_nodal][0], 1, u_nodal_values[index_nodal]))


    # Save solution in VTK format
    file = File("poisson.pvd")
    file << u
    
    # Compute errors
    print("Computing the error")
    

    uexact = eval(obj_expression_u.func_value_u())
    
    
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
    
    
    
    array_error[i][0]=l2_error_u
    array_error[i][1]=h1_semi_error_u
    array_error[i][2]=h2_semi_error_u
    
    if i>0 and i<=total_refinements:
        for k in range(3):
            array_order_of_convergence[i-1][k] = math.log(array_error[i-1][k]/array_error[i][k],10)/math.log(2,10)

    time_cpu_elapsed = time.time() - t
    
    array_time_cpu[i]=time_cpu_elapsed
    
    print("time elapsed:", "{:6.4f}".format(time_cpu_elapsed))
    
    file_error.write("%s %s %0.2e %0.2e %0.2e %0.2e\n" %(current_refinement_level, n_dofs, l2_error_u, h1_semi_error_u, h2_semi_error_u, time_cpu_elapsed))
    
    current_refinement_level += 1
    i += 1
    n_rect_cell_one_direction = n_rect_cell_one_direction*2

print()
print("====================================")
print("SUMMARY")
np.set_printoptions(formatter={'float': '{: 0.2e}'.format})

print("@error")
for line in array_error:
    print(line)

print("@order of convergence")
np.set_printoptions(formatter={'float': '{: 0.2f}'.format})
for line in array_order_of_convergence:
    print(line)
    
print("@cpu time")
np.set_printoptions(formatter={'float': '{: 0.2e}'.format})
for line in array_time_cpu:
    print(line)


print()
    
file_error.close() 


# Plot solution
#import matplotlib.pyplot as plt
#plot(u)
#plt.show()
