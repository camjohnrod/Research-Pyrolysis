import numpy as np
import matplotlib.pyplot as plt
from   tqdm import tqdm
from tqdm.auto import trange
import os

import dolfinx_mpc.utils
from   dolfinx import fem, mesh, io, default_scalar_type
from   dolfinx.io import XDMFFile
from   dolfinx_mpc import LinearProblem as MPCLinearProblem
from   dolfinx.fem.petsc import NonlinearProblem, LinearProblem
from   dolfinx.nls.petsc import NewtonSolver
from   mpi4py import MPI
import ufl

from unit_cell import solve_unit_cell


##======================================##
##=============== INPUTS ===============##
##======================================##


is_homogenize = True
save_every = 2

t0                    = 0.0
tf                    = 1
temp_ramp_duration    = tf
num_timesteps         = int(10)
dt                    = tf / num_timesteps

k_poly         = 12.6
k_cer          = 120.0
k_fib          = 54

cp_poly        = 1170.0
cp_cer         = 750.0
cp_fib         = 879

alpha_mat      = 3.95e-6
alpha_fib      = alpha_mat #-0.64e-6

rho_poly       = 1150.0    # updated
rho_cer        = 2450.0    # updated
rho_fib        = 1780     

E_poly         = 4.94e9    # updated
E_cer          = 206.18e9  # updated
E_fib          = 264.5e9

# the Zhang paper didn't have poisson's ratios, so these are assumed

nu_poly        = 0.3
nu_cer         = 0.14
nu_fib         = 0.14 

initial_temp   = 0.0
final_temp     = 1200.0

length         = 24e-6
width          = 24e-6
height         = 24e-6

# these values are also assumed 

vf_cer_0       = 0.0               # initial ceramic volume fraction (=0.0 for the first pyrolysis cycle)
vf_poly_0      = 1.0               # initial polymer volume fraction (=1.0 for the first pyrolysis cycle)
a              = 0.05              # "void formation ratio of the precursor", value is not mentioned in the Zhang paper

# taken from the Zhang paper 

n              = 2
A_factor       = 1.5795 / (60)
E_a            = 18216.0
R_gas          = 8.3145


##======================================##
##========== DEFINE THE MESH ===========##
##======================================##


with XDMFFile(MPI.COMM_WORLD, "mesh/rve_3D.xdmf", "r") as xdmf:
    domain    = xdmf.read_mesh(name="Grid")
    cell_tags = xdmf.read_meshtags(domain, name="Grid")      


##======================================##
##===== DEFINE MATERIAL PROPERTIES =====##
##======================================##


S_material_prop = fem.functionspace(domain, ("Lagrange", 1))

r_func     = fem.Function(S_material_prop, name="r")
k_func     = fem.Function(S_material_prop, name="k")
cp_func    = fem.Function(S_material_prop, name="cp")
alpha_func = fem.Function(S_material_prop, name="alpha")
rho_func   = fem.Function(S_material_prop, name="rho")
E_func     = fem.Function(S_material_prop, name="E")
nu_func    = fem.Function(S_material_prop, name="nu")
mu_func    = fem.Function(S_material_prop, name="mu")
lam_func   = fem.Function(S_material_prop, name="lam")

def get_vf_cer(r, a, vf_poly_0):
    return (1 - a) * r * vf_poly_0

def get_vf_void(r, a, vf_poly_0):
    return a * r * vf_poly_0

def get_vf_poly(vf_cer, vf_cer_0, vf_void):
    return 1 - vf_cer - vf_cer_0 - vf_void 

def rule_of_mixtures(prop_poly, vf_poly, prop_cer, vf_cer):
    return prop_poly * vf_poly + prop_cer * vf_cer

def update_matrix_material_properties(r_old, u_temp_prev):

    r_new     = A_factor * (np.exp(-E_a / (R_gas * (u_temp_prev + 273.15)))) * (1 - r_old) ** n * dt + r_old
    
    vf_cer    = get_vf_cer(r_new, a, vf_poly_0)
    vf_void   = get_vf_void(r_new, a, vf_poly_0)
    vf_poly   = get_vf_poly(vf_cer, vf_cer_0, vf_void)

    k_val     = rule_of_mixtures(k_poly, vf_poly, k_cer, vf_cer)
    cp_val    = rule_of_mixtures(cp_poly, vf_poly, cp_cer, vf_cer) # is this the right way to calculate cp
    alpha_val = rule_of_mixtures(alpha_mat, vf_poly, alpha_mat, vf_cer)
    rho_val   = rule_of_mixtures(rho_poly, vf_poly, rho_cer, vf_cer)
    E_val     = rule_of_mixtures(E_poly, vf_poly, E_cer, vf_cer)
    nu_val    = rule_of_mixtures(nu_poly, vf_poly, nu_cer, vf_cer)
    mu_val    = E_val / (2 * (1 + nu_val))
    lam_val   = E_val * nu_val / ((1 + nu_val) * (1 - 2 * nu_val))

    r_func.x.array[:]     = r_new
    k_func.x.array[:]     = k_val
    cp_func.x.array[:]    = cp_val
    alpha_func.x.array[:] = alpha_val
    rho_func.x.array[:]   = rho_val
    E_func.x.array[:]     = E_val
    nu_func.x.array[:]    = nu_val
    mu_func.x.array[:]    = mu_val
    lam_func.x.array[:]   = lam_val

    r_func.x.scatter_forward()
    cp_func.x.scatter_forward()
    k_func.x.scatter_forward()
    alpha_func.x.scatter_forward()
    rho_func.x.scatter_forward()
    E_func.x.scatter_forward()
    nu_func.x.scatter_forward()
    mu_func.x.scatter_forward()
    lam_func.x.scatter_forward()

mu_fib  = E_fib / (2 * (1 + nu_fib))
lam_fib = E_fib * nu_fib / ((1 + nu_fib) * (1 - 2 * nu_fib))

material_properties = { 
    1: (mu_fib, lam_fib, alpha_fib, k_fib, cp_fib, rho_fib),
    2: (mu_func, lam_func, alpha_func, k_func, cp_func, rho_func)
}


##==================================================##
##======= DEFINE TEMPERATURE FUNCTION SPACES =======##
##==================================================##


S_temp = fem.functionspace(domain, ("Lagrange", 1))
S_disp = fem.functionspace(domain, ("Lagrange", 1, (domain.geometry.dim, )))

u_temp_current   = fem.Function(S_temp)
u_temp_prev      = fem.Function(S_temp)
u_temp_prev.name = "Temperature"
v_temp_current   = ufl.TestFunction(S_temp)


##==============================================##
##=== CALCULATE STARTING MATERIAL PROPERTIES ===##
##==============================================##


update_matrix_material_properties(r_func.x.array[:], u_temp_prev.x.array[:])

if is_homogenize:
    stiffness_tensor_homogenized = solve_unit_cell(r_func.x.array[:])


##======================================##
##==== DEFINE EACH BOUNDARY SURFACE ====##
##======================================##


def left(x):
    return (np.isclose(x[0], 0))      # & (x[1] > 0) & (x[1] < width) & (x[2] > 0) & (x[2] < height))    
def right(x):
    return (np.isclose(x[0], length)) # & (x[1] > 0) & (x[1] < width) & (x[2] > 0) & (x[2] < height))
def front(x):
    return (np.isclose(x[1], 0))      # & (x[0] > 0) & (x[0] < length) & (x[2] > 0) & (x[2] < height))
def back(x):
    return (np.isclose(x[1], width))  # & (x[0] > 0) & (x[0] < length) & (x[2] > 0) & (x[2] < height))
def bottom(x):
    return (np.isclose(x[2], 0))      # & (x[0] > 0) & (x[0] < length) & (x[1] > 0) & (x[1] < width))
def top(x):
    return (np.isclose(x[2], height)) # & (x[0] > 0) & (x[0] < length) & (x[1] > 0) & (x[1] < width))


##======================================##
##======== BOUNDARY CONDITIONS =========##
##======================================##


def fixed_temp_dof(x):
    return np.isclose(x[0], 0.0) | np.isclose(x[0], length) | np.isclose(x[1], 0.0) | np.isclose(x[1], width)

def fixed_disp_dof(x):
    return np.isclose(x[2], 0.0)

fdim_plane = 2

fixed_facets_temp = mesh.locate_entities_boundary(domain, fdim_plane, fixed_temp_dof)
fixed_dofs_temp   = fem.locate_dofs_topological(S_temp, fdim_plane, fixed_facets_temp)
boundary_temp     = fem.Constant(domain, default_scalar_type(initial_temp))
bcs_temp          = [fem.dirichletbc(boundary_temp, fixed_dofs_temp, S_temp)]

fixed_facets_disp = mesh.locate_entities_boundary(domain, fdim_plane, fixed_disp_dof)
fixed_dofs_disp   = fem.locate_dofs_topological(S_disp, fdim_plane, fixed_facets_disp)
bcs_disp = fem.dirichletbc(np.array([0, 0, 0], dtype=default_scalar_type), fixed_dofs_disp, S_disp)


##===========================================##
##=== DEFINE TEMPERATURE VARIATIONAL FORM ===##
##===========================================##


u_temp_prev.interpolate(lambda x: np.full(x.shape[1], initial_temp, dtype=default_scalar_type)) # is interpolate necessary

def epsilon(u):
    return ufl.grad(u)

dx = ufl.Measure("dx", domain=domain, subdomain_data=cell_tags)

R_temp = 0.0
for tag, (_, _, _, k, cp, rho) in material_properties.items():

    R_temp += rho * cp * (u_temp_current - u_temp_prev) / dt * v_temp_current * dx(tag) \
           + ufl.dot(k * epsilon(u_temp_current), epsilon(v_temp_current)) * dx(tag)

J_temp = ufl.derivative(R_temp, u_temp_current)

problem_temp = NonlinearProblem(R_temp, u_temp_current, bcs_temp, J_temp)
solver_temp = NewtonSolver(domain.comm, problem_temp)
solver_temp.atol = 1e-6
solver_temp.rtol = 1e-6
solver_temp.max_it = 50
solver_temp.convergence_criterion = "incremental"


##===========================================##
##=== DEFINE DISPLACEMENT FUNCTION SPACES ===##
##===========================================##


u_disp_current = ufl.TrialFunction(S_disp)
v_disp_current = ufl.TestFunction(S_disp)

S_constant = fem.functionspace(domain, ("DG", 0))

vm_stress_current = fem.Function(S_constant) 
vm_stress_current.name = "von Mises Stress"

u_disp_current_store = fem.Function(S_disp) 
u_disp_current_store.name = "Displacement"
u_disp_prev = fem.Function(S_disp)


##================================================================##
##=== DEFINE THE (LINEAR) VARIATIONAL PROBLEM FOR DISPLACEMENT ===##
##================================================================##


dim_disp = domain.geometry.dim
I = ufl.variable(ufl.Identity(dim_disp))

def get_stiffness_tensor(mu, lam):
    if is_homogenize:
        C = stiffness_tensor_homogenized
    else:
        C = ufl.as_matrix([[lam + 2 * mu,     lam,           lam,           0,              0,              0],
                            [lam,           lam + 2 * mu,     lam,           0,              0,              0],
                            [lam,               lam,       lam + 2 * mu,     0,              0,              0],
                            [0,                 0,              0,          mu,              0,              0],
                            [0,                 0,              0,           0,             mu,              0],
                            [0,                 0,              0,           0,              0,             mu]])
    return C

def get_voigt(matrix):
    return ufl.as_vector([    matrix[0, 0],   # ₁₁
                              matrix[1, 1],   # ₂₂
                              matrix[2, 2],   # ₃₃
                          2 * matrix[1, 2],   # ₂₃ 
                          2 * matrix[0, 2],   # ₁₃ 
                          2 * matrix[0, 1]    # ₁₂ 
                        ])

def epsilon_sym(u):
    epsilon = ufl.sym(ufl.grad(u))
    return get_voigt(epsilon)

def epsilon_volume(r):
    beta = -0.3 * r ** 2 + 0.22 * r - 0.17
    return get_voigt(beta * r * I)

def epsilon_thermal(alpha, delta_temp):
    return get_voigt(alpha * delta_temp * I)

def P_tot(u, mu, lam):
    return ufl.dot(get_stiffness_tensor(mu, lam), epsilon_sym(u))

def P_volume(mu, lam, r):
    return ufl.dot(get_stiffness_tensor(mu, lam), epsilon_volume(r))

def P_thermal(mu, lam, alpha, delta_temp):
    return ufl.dot(get_stiffness_tensor(mu, lam), epsilon_thermal(alpha, delta_temp))

delta_temp = u_temp_prev - initial_temp

a_disp = 0.0
L_disp = 0.0

for tag, (mu, lam, alpha, _, _, _) in material_properties.items():
    if tag == 2:
        a_disp += ufl.inner(epsilon_sym(v_disp_current), P_tot(u_disp_current, mu, lam)) * dx(tag)
        L_disp += ufl.inner(epsilon_sym(v_disp_current), P_thermal(mu, lam, alpha, delta_temp)) * dx(tag)
        # L_disp += ufl.inner(epsilon_sym(v_disp_current), P_volume(mu, lam, r_func)) * dx(tag)
    else:
        a_disp += ufl.inner(epsilon_sym(v_disp_current), P_tot(u_disp_current, mu, lam)) * dx(tag)
        L_disp += ufl.inner(epsilon_sym(v_disp_current), P_thermal(mu, lam, alpha, delta_temp)) * dx(tag)

problem_disp = LinearProblem(a_disp, L_disp, bcs=[bcs_disp])


##=================================================##
##=== DEFINE THE PROJECTION FOR OTHER VARIABLES ===##
##=================================================##


phi    = ufl.TrialFunction(S_constant)
psi    = ufl.TestFunction(S_constant)
a_proj = ufl.inner(phi, psi) * dx

L_proj_von_mises = 0.0
for tag, (mu, lam, alpha, _, _, _) in material_properties.items():
    if tag == 2:
        tot_stress_expr = P_tot(u_disp_current_store, mu, lam) - P_thermal(mu, lam, alpha, delta_temp) - P_volume(mu, lam, r_func)
    else:
        tot_stress_expr = P_tot(u_disp_current_store, mu, lam) - P_thermal(mu, lam, alpha, delta_temp)       
        
    von_mises_expr = ufl.sqrt(0.5 * ((tot_stress_expr[0] - tot_stress_expr[1])**2 +
                                     (tot_stress_expr[1] - tot_stress_expr[2])**2 +
                                     (tot_stress_expr[2] - tot_stress_expr[0])**2 +
                                     6 * (tot_stress_expr[3]**2 + tot_stress_expr[4]**2 + tot_stress_expr[5]**2)))
    L_proj_von_mises += von_mises_expr * psi * dx(tag)

problem_stress_projection = LinearProblem(a_proj, L_proj_von_mises, bcs=[])


##=================================================================##
##=== INITIALIZE THE .xdmf OUTPUT FILE AND INITIAL (t0) OUTPUTS ===##
##=================================================================##


if not os.path.exists("results"):
    os.makedirs("results")

xdmf = io.XDMFFile(domain.comm, "results/results_3D.xdmf", "w")
xdmf.write_mesh(domain)

xdmf.write_function(u_temp_prev, t0)
xdmf.write_function(u_disp_current_store, t0)
xdmf.write_function(vm_stress_current, t0)
xdmf.write_function(r_func, t0)
xdmf.write_function(E_func, t0)


##======================================##
##============ TIME STEPPING ===========##
##======================================##


r_middle_point_values = []
temp_middle_point_values = []

t = t0 + dt
print('\n')
for i in trange(num_timesteps, colour="blue", desc="  Time Stepping", position=0, ncols=100):
    
    r_middle_point    = r_func.x.array[len(r_func.x.array) // 2]
    temp_middle_point = u_temp_prev.x.array[len(u_temp_prev.x.array) // 2]

    r_middle_point_values.append(r_middle_point)
    temp_middle_point_values.append(temp_middle_point)

    ramp_param = min(max(t / temp_ramp_duration, 0.0), 1.0)
    temp_bc    = initial_temp + ramp_param * (final_temp - initial_temp)
    boundary_temp.value = default_scalar_type(temp_bc)

    num_its_temp, converged_temp = solver_temp.solve(u_temp_current)
    if not converged_temp:
        raise RuntimeError(f"Newton failed to converge at timestep {i+1}")
    assert (converged_temp)
    u_temp_current.x.scatter_forward()

    u_temp_prev.x.array[:] = u_temp_current.x.array[:]
    u_temp_prev.x.scatter_forward()

    u_disp_sol = problem_disp.solve()
    u_disp_sol.x.scatter_forward()

    u_disp_prev.x.array[:] = u_disp_current_store.x.array[:]
    u_disp_prev.x.scatter_forward()
    u_disp_current_store.x.array[:] = u_disp_sol.x.array[:]
    u_disp_current_store.x.scatter_forward()

    if i % save_every == 0 or i == num_timesteps - 1:
        xdmf.write_function(u_temp_prev, t)
        xdmf.write_function(u_disp_current_store, t)

        vm_stress_current = problem_stress_projection.solve() 
        vm_stress_current.name = "von Mises Stress"
        vm_stress_current.x.scatter_forward()
        xdmf.write_function(vm_stress_current, t)

        xdmf.write_function(r_func, t)
        xdmf.write_function(E_func, t)

    update_matrix_material_properties(r_func.x.array[:], u_temp_prev.x.array[:])

    if is_homogenize:
        stiffness_tensor_homogenized = solve_unit_cell(r_func.x.array[:])

    t += dt
print('\n')

xdmf.close()

plt.figure()
plt.plot(temp_middle_point_values, r_middle_point_values, marker='o', color='r')
plt.xlabel('Temperature')
plt.ylabel('r')
plt.title('Ceramization Degree (r) vs Temperature', fontsize=16)
plt.grid(True)
plt.savefig('results/r_vs_temperature.png')