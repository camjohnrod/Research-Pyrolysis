import numpy as np
import matplotlib.pyplot as plt
import os
from   tqdm import tqdm
from   tqdm.auto import trange
from   dataclasses import dataclass

import ufl
import dolfinx_mpc.utils
from   dolfinx import fem, mesh, io, default_scalar_type
from   dolfinx.io import XDMFFile
from   dolfinx_mpc import LinearProblem as MPCLinearProblem
from   dolfinx.fem.petsc import NonlinearProblem, LinearProblem
from   dolfinx.nls.petsc import NewtonSolver
from   mpi4py import MPI

from unit_cell_mpc import get_mpc
from unit_cell_solve import solve_unit_cell


if not os.path.exists("results"):
    os.makedirs("results")

if not os.path.exists("mesh"):
    os.makedirs("mesh")


##======================================##
##=============== INPUTS ===============##
##======================================##


save_every = 5

vf_fib = 0.267 # hard coded

t0                        = 0.0
temp_ramp_duration        = 3 * 60 * 60
temp_long_hold_duration   = 3 * 60 * 60
temp_short_hold_duration  = 1 * 60 * 60
num_cycles                = 1
total_cycle_length        = (2 * temp_ramp_duration + temp_long_hold_duration + temp_short_hold_duration)
tf                        = num_cycles * total_cycle_length
num_timesteps             = int(tf / 60 / 5)
dt                        = tf / num_timesteps

k_poly         = 12.6
k_cer          = 120.0
k_fib          = 54

cp_poly        = 1170.0
cp_cer         = 750.0
cp_fib         = 879

alpha_mat      = 3.95e-6
alpha_fib      = -0.64e-6

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

# these values are also assumed 

vf_cer_0       = 0.0          # initial ceramic volume fraction (=0.0 for the first pyrolysis cycle)
vf_poly_0      = 1.0          # initial polymer volume fraction (=1.0 for the first pyrolysis cycle)
a              = 0.2         # "void formation ratio of the precursor", value is not mentioned in the Zhang paper

# taken from the Zhang paper 

n              = 2
A_factor       = 1.5795 / (60)
E_a            = 18216.0
R_gas          = 8.3145


##======================================##
##========== DEFINE THE MESH ===========##
##======================================##


with XDMFFile(MPI.COMM_WORLD, "mesh/domain_3D.xdmf", "r") as xdmf:
    domain    = xdmf.read_mesh(name="Grid")
    cell_tags = xdmf.read_meshtags(domain, name="Grid")      
    

##==============================================##
##=========== DEFINE FUNCTION SPACES ===========##
##==============================================##


S_temp = fem.functionspace(domain, ("Lagrange", 1))
S_disp = fem.functionspace(domain, ("Lagrange", 1, (domain.geometry.dim, )))
S_constant = fem.functionspace(domain, ("DG", 0))

u_temp_current   = fem.Function(S_temp)
u_temp_prev      = fem.Function(S_temp)
u_temp_prev.name = "Temperature"
v_temp_current   = ufl.TestFunction(S_temp)

u_disp_current = ufl.TrialFunction(S_disp)
v_disp_current = ufl.TestFunction(S_disp)

u_disp_current_store = fem.Function(S_disp) 
u_disp_current_store.name = "Displacement"
u_disp_prev = fem.Function(S_disp)

vm_stress_current = fem.Function(S_constant) 
vm_stress_current.name = "von Mises Stress"


##======================================##
##===== DEFINE MATERIAL PROPERTIES =====##
##======================================##


@dataclass(frozen=True)
class MaterialConstants:
    k: float
    cp: float
    alpha: float
    rho: float
    E: float
    nu: float

    @property
    def mu(self) -> float:
        return self.E / (2.0 * (1.0 + self.nu))

    @property
    def lam(self) -> float:
        return self.E * self.nu / ((1.0 + self.nu) * (1.0 - 2.0 * self.nu))

fiber   = MaterialConstants(k_fib, cp_fib, alpha_fib, rho_fib, E_fib, nu_fib)
polymer = MaterialConstants(k_poly, cp_poly, alpha_mat, rho_poly, E_poly, nu_poly)
ceramic = MaterialConstants(k_cer, cp_cer, alpha_mat, rho_cer, E_cer, nu_cer)

class MaterialState:
    def __init__(
        self,
        domain,
        fiber: MaterialConstants,
        polymer: MaterialConstants,
        ceramic: MaterialConstants,
        vf_fib: float,
    ):
        self.domain = domain
        self.fiber = fiber
        self.polymer = polymer
        self.ceramic = ceramic
        self.vf_fib = vf_fib

        self.S = fem.functionspace(domain, ("Lagrange", 1))

        self.r_old = fem.Function(self.S, name="r_old")
        self.r_new = fem.Function(self.S, name="r_new")
        self.k     = fem.Function(self.S, name="k")
        self.cp    = fem.Function(self.S, name="cp")
        self.alpha = fem.Function(self.S, name="alpha")
        self.rho   = fem.Function(self.S, name="rho")
        self.E     = fem.Function(self.S, name="E")
        self.nu    = fem.Function(self.S, name="nu")
        self.mu    = fem.Function(self.S, name="mu")
        self.lam   = fem.Function(self.S, name="lam")
        
        self.vf_cer  = 0
        self.vf_void = 0
        self.vf_poly = 0

    def rule_of_mixtures_matrix(self, prop_poly, vf_poly, prop_cer, vf_cer):
        return prop_poly * vf_poly + prop_cer * vf_cer

    def rule_of_mixtures_total(self, prop_matrix, prop_fib):
        vf_matrix = 1.0 - self.vf_fib
        return vf_matrix * prop_matrix + self.vf_fib * prop_fib
    
    def update(self, r_old, temp, dt, vf_poly_0, vf_cer_0):

        r_step = (A_factor * np.exp(-E_a / (R_gas * (temp + 273.15))) * (1.0 - r_old) ** n * dt + r_old)
        if (r_step >= r_old).all():
            r_new = r_step

        self.vf_cer  = (1 - a) * r_new * vf_poly_0 + vf_cer_0
        self.vf_void = a * r_new * vf_poly_0
        self.vf_poly = 1.0 - self.vf_cer - self.vf_void

        k_m      = self.rule_of_mixtures_matrix(self.polymer.k, self.vf_poly, self.ceramic.k, self.vf_cer)
        cp_m     = self.rule_of_mixtures_matrix(self.polymer.cp, self.vf_poly, self.ceramic.cp, self.vf_cer)
        alpha_m  = self.rule_of_mixtures_matrix(self.polymer.alpha, self.vf_poly, self.ceramic.alpha, self.vf_cer)
        rho_m    = self.rule_of_mixtures_matrix(self.polymer.rho, self.vf_poly, self.ceramic.rho, self.vf_cer)
        E_m      = self.rule_of_mixtures_matrix(self.polymer.E, self.vf_poly, self.ceramic.E, self.vf_cer)
        nu_m     = self.rule_of_mixtures_matrix(self.polymer.nu, self.vf_poly, self.ceramic.nu, self.vf_cer)

        self.r_old.x.array[:] = r_old
        self.r_new.x.array[:] = r_new
        self.k.x.array[:]     = self.rule_of_mixtures_total(k_m,  self.fiber.k)
        self.cp.x.array[:]    = self.rule_of_mixtures_total(cp_m, self.fiber.cp)
        self.alpha.x.array[:] = self.rule_of_mixtures_total(alpha_m,  self.fiber.alpha)
        self.rho.x.array[:]   = self.rule_of_mixtures_total(rho_m,self.fiber.rho)

        self.E.x.array[:]   = E_m
        self.nu.x.array[:]  = nu_m
        self.mu.x.array[:]  = E_m / (2.0 * (1.0 + nu_m))
        self.lam.x.array[:] = E_m * nu_m / ((1.0 + nu_m) * (1.0 - 2.0 * nu_m))

        for f in (self.r_new, self.k, self.cp, self.alpha, self.rho,
                self.E, self.nu, self.mu, self.lam):
            f.x.scatter_forward()

        return r_new

material_state = MaterialState(domain, fiber, polymer, ceramic, vf_fib)


##==============================================##
##=== CALCULATE STARTING MATERIAL PROPERTIES ===##
##==============================================##


with XDMFFile(MPI.COMM_WORLD, "mesh/rve_3D.xdmf", "r") as xdmf:
    domain_unit_cell    = xdmf.read_mesh(name="Grid")
    cell_tags_unit_cell = xdmf.read_meshtags(domain_unit_cell, name="Grid")      

u_temp_prev.interpolate(lambda x: np.full(x.shape[1], initial_temp, dtype=default_scalar_type)) # is interpolate necessary
delta_temp = u_temp_prev - initial_temp

material_state.update(material_state.r_new.x.array, u_temp_prev.x.array, dt, vf_poly_0, vf_cer_0)
mpc, bcs_disp_unit_cell = get_mpc(domain_unit_cell)

previous_beta = fem.Constant(domain_unit_cell, 0.0)
stiffness_tensor_homogenized = fem.Constant(domain_unit_cell, np.zeros((6, 6), dtype=default_scalar_type))
eigenstrain_homogenized_previous = fem.Constant(domain_unit_cell, np.zeros(6, dtype=default_scalar_type))
eigenstrain_homogenized = fem.Constant(domain_unit_cell, np.zeros(6, dtype=default_scalar_type))
solve_unit_cell(domain_unit_cell, cell_tags_unit_cell, material_state, mpc, bcs_disp_unit_cell, u_temp_prev, previous_beta, stiffness_tensor_homogenized, eigenstrain_homogenized)


##======================================##
##======== BOUNDARY CONDITIONS =========##
##======================================##


def is_boundary(x):
    return ((x[0] <= -0.013) & (x[1] <= 0.033))

def all_surfaces(x):
    return np.full(x.shape[1], True, dtype=bool)

fdim_plane = domain.topology.dim - 1

disp_facets         = mesh.locate_entities_boundary(domain, fdim_plane, is_boundary)
temp_facets         = mesh.locate_entities_boundary(domain, fdim_plane, all_surfaces)
fixed_dofs_disp   = fem.locate_dofs_topological(S_disp, fdim_plane, disp_facets)
fixed_dofs_temp   = fem.locate_dofs_topological(S_temp, fdim_plane, temp_facets)

boundary_temp   = fem.Constant(domain, default_scalar_type(initial_temp))
bcs_temp        = [fem.dirichletbc(boundary_temp, fixed_dofs_temp, S_temp)]

bcs_disp = fem.dirichletbc(fem.Constant(domain, np.array([0.0, 0.0, 0.0], dtype=default_scalar_type)), fixed_dofs_disp, S_disp)


##==================================================##
##=== DEFINE THE VARIATIONAL FORM FOR TEMPERATURE===##
##==================================================##


def epsilon(u):
    return ufl.grad(u)

dx = ufl.Measure("dx", domain=domain)

R_temp = material_state.rho * material_state.cp * (u_temp_current - u_temp_prev) / dt * v_temp_current * dx \
        + ufl.dot(material_state.k * epsilon(u_temp_current), epsilon(v_temp_current)) * dx

J_temp = ufl.derivative(R_temp, u_temp_current)

problem_temp = NonlinearProblem(R_temp, u_temp_current, bcs_temp, J_temp)
solver_temp = NewtonSolver(domain.comm, problem_temp)
solver_temp.atol = 1e-6
solver_temp.rtol = 1e-6
solver_temp.max_it = 50
solver_temp.convergence_criterion = "incremental"


##=============================================================##
##=== DEFINE THE (LINEAR) VARIATIONAL FORM FOR DISPLACEMENT ===##
##=============================================================##


dim_disp = domain.geometry.dim
I = ufl.variable(ufl.Identity(dim_disp))

def get_voigt(tensor):
    return ufl.as_vector([    tensor[0, 0],   # ₁₁
                              tensor[1, 1],   # ₂₂
                              tensor[2, 2],   # ₃₃
                          2 * tensor[1, 2],   # ₂₃ 
                          2 * tensor[0, 2],   # ₁₃ 
                          2 * tensor[0, 1]    # ₁₂ 
                        ])

def epsilon_sym(u):
    epsilon = ufl.sym(ufl.grad(u))
    return get_voigt(epsilon)

def P_tot(u, C):
    return ufl.dot(C, epsilon_sym(u))

def P_eigenstrain(eig, C):
    return ufl.dot(C, eig)

a_disp = ufl.inner(epsilon_sym(v_disp_current), P_tot(u_disp_current, stiffness_tensor_homogenized)) * dx
L_disp = ufl.inner(epsilon_sym(v_disp_current), P_eigenstrain(eigenstrain_homogenized + eigenstrain_homogenized_previous, stiffness_tensor_homogenized)) * dx
problem_disp = LinearProblem(a_disp, L_disp, bcs=[bcs_disp])


##=================================================##
##=== DEFINE THE PROJECTION FOR OTHER VARIABLES ===##
##=================================================##


phi    = ufl.TrialFunction(S_constant)
psi    = ufl.TestFunction(S_constant)
a_proj = ufl.inner(phi, psi) * dx

tot_stress_expr = P_tot(u_disp_current_store, stiffness_tensor_homogenized) - P_eigenstrain(eigenstrain_homogenized + eigenstrain_homogenized_previous, stiffness_tensor_homogenized)       
von_mises_expr = ufl.sqrt(0.5 * ((tot_stress_expr[0] - tot_stress_expr[1])**2 +
                                 (tot_stress_expr[1] - tot_stress_expr[2])**2 +
                                 (tot_stress_expr[2] - tot_stress_expr[0])**2 +
                             6 * (tot_stress_expr[3]**2 + tot_stress_expr[4]**2 + tot_stress_expr[5]**2)))
L_proj_von_mises = von_mises_expr * psi * dx
problem_stress_projection = LinearProblem(a_proj, L_proj_von_mises, bcs=[])


##=================================================================##
##=== INITIALIZE THE .xdmf OUTPUT FILE AND INITIAL (t0) OUTPUTS ===##
##=================================================================##


xdmf = io.XDMFFile(domain.comm, "results/results_3D.xdmf", "w")
xdmf.write_mesh(domain)

# xdmf.write_function(u_temp_prev, t0)
# xdmf.write_function(u_disp_current_store, t0)
# xdmf.write_function(vm_stress_current, t0)
# xdmf.write_function(material_state.r, t0)
# xdmf.write_function(material_state.E, t0)


##======================================##
##============ TIME STEPPING ===========##
##======================================##


r_avg_values = [np.mean(material_state.r_new.x.array[:])]
temp_avg_values = [np.mean(u_temp_prev.x.array[:])]
vf_poly_avg_values = [np.mean(material_state.vf_poly)]
vf_cer_avg_values = [np.mean(material_state.vf_cer)]
vf_void_avg_values = [np.mean(material_state.vf_void)]
E_avg_values = [1 / np.linalg.inv(stiffness_tensor_homogenized.value)[0,0]]
time_vector = np.zeros(num_timesteps + 1)

t = t0 + dt

old_cycle = 0

for i in trange(num_timesteps, colour="blue", desc="  Time Stepping", position=0, bar_format='{l_bar}{bar:30}{r_bar}', total=num_timesteps):

    ##==========================================##
    ##=== SOLVE AND OUTPUT SOLUTION TO .xdmf ===##
    ##==========================================##

    current_cycle = np.floor(t / total_cycle_length)

    ramp_up_param = min((t - current_cycle * total_cycle_length) / temp_ramp_duration, 1.0)
    ramp_down_param = max(min((max((t - current_cycle * total_cycle_length), (temp_ramp_duration + temp_long_hold_duration)) \
                        - (temp_ramp_duration + temp_long_hold_duration))  / temp_ramp_duration, 1.0), 0.0)

    temp_bc    = initial_temp + ramp_up_param * (final_temp - initial_temp) - ramp_down_param * (final_temp - initial_temp)
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

        xdmf.write_function(material_state.r_new, t)
        xdmf.write_function(material_state.E, t)

    if current_cycle != old_cycle:
        vf_poly_0 = material_state.vf_void
        vf_cer_0  = material_state.vf_cer
        material_state = MaterialState(domain, fiber, polymer, ceramic, vf_fib)
        material_state.update(material_state.r_new.x.array, u_temp_prev.x.array, dt, vf_poly_0, vf_cer_0)
        old_cycle = current_cycle
    else:
        material_state.update(material_state.r_new.x.array, u_temp_prev.x.array, dt, vf_poly_0, vf_cer_0)
    solve_unit_cell(domain_unit_cell, cell_tags_unit_cell, material_state, mpc, bcs_disp_unit_cell, u_temp_prev, previous_beta, stiffness_tensor_homogenized, eigenstrain_homogenized)

    ##=================================##
    ##=== CALCULATE MEAN QUANTITIES ===##
    ##=================================##

    time_vector[i + 1] = t / 60 / 60
    r_avg = np.mean(material_state.r_new.x.array[:])
    temp_avg = np.mean(u_temp_prev.x.array[:])
    vf_poly_avg = np.mean(material_state.vf_poly)
    vf_cer_avg = np.mean(material_state.vf_cer)
    vf_void_avg = np.mean(material_state.vf_void)

    r_avg_values.append(r_avg)
    temp_avg_values.append(temp_avg)
    vf_poly_avg_values.append(vf_poly_avg)
    vf_cer_avg_values.append(vf_cer_avg)
    vf_void_avg_values.append(vf_void_avg)
    E_avg_values.append(1 / np.linalg.inv(stiffness_tensor_homogenized.value)[0,0])

    t += dt

xdmf.close()


##===========================================##
##=== PLOT QUANTITIES AT THE CENTER POINT ===##
##===========================================##


plt.figure(1)
plt.plot(time_vector[:-2], r_avg_values[:-2], marker='o', color='g')
plt.xlabel('Time (hr)', fontsize=14)
plt.ylabel('r', fontsize=14)
plt.title('Ceramization Degree vs Time', fontsize=16)
plt.grid(True)
plt.savefig('results/r_vs_time.png')

plt.figure(2)
plt.plot(time_vector, temp_avg_values, marker='o', color='r')
plt.xlabel('Time (hr)', fontsize=14)
plt.ylabel('Temperature (C)', fontsize=14)
plt.title('Temperature vs Time', fontsize=16)
plt.grid(True)
plt.savefig('results/temp_vs_time.png')

plt.figure(3)
plt.plot(time_vector, vf_poly_avg_values, color='green', linewidth = 3, label='Polymer Volume Fraction')
plt.plot(time_vector, vf_cer_avg_values, color='orange', linewidth = 3, label='Ceramic Volume Fraction')
plt.plot(time_vector, vf_void_avg_values, color='gray', linewidth = 3, label='Void Volume Fraction')
plt.xlabel('Time (hr)', fontsize=14)
plt.ylabel('Volume Fraction', fontsize=14)
plt.title('Volume Fractions vs Time', fontsize=16)
plt.legend(fontsize=12)
plt.grid(True)
plt.savefig('results/volume_fractions_vs_time.png')

plt.figure(4)
#make a single plot with E vs t on the left axis and temp vs t on the right axis

ax1 = plt.gca()
ax2 = ax1.twinx()

ax1.plot(time_vector, E_avg_values, color='blue', linewidth=3, label='Elastic Modulus')
ax2.plot(time_vector, temp_avg_values, color='red', linewidth=3, label='Temperature')

ax1.set_xlabel('Time (hr)', fontsize=14)
ax1.set_ylabel('Elastic Modulus (Pa)', fontsize=14)
ax2.set_ylabel('Temperature (C)', fontsize=14)

plt.title('Elastic Modulus and Temperature vs Time', fontsize=16)
plt.grid(True)
plt.savefig('results/E_and_temp_vs_time.png')

plt.figure(5)
ramp_up_end_index = int(temp_ramp_duration / dt)
plt.plot(temp_avg_values[0:ramp_up_end_index], r_avg_values[0:ramp_up_end_index], marker='o', color='orange')
plt.xlabel('Temperature (C)', fontsize=14)
plt.ylabel('r', fontsize=14)
plt.title('Ceramization Degree vs Temperature', fontsize=16)
plt.grid(True)
plt.savefig('results/r_vs_temp.png')