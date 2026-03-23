import numpy as np
import matplotlib.pyplot as plt
import os
import yaml
from   tqdm import tqdm
from   tqdm.auto import trange
from   dataclasses import dataclass

import ufl
import dolfinx_mpc.utils
from   dolfinx import fem, mesh, io, default_scalar_type
from   dolfinx.io import XDMFFile
from   petsc4py import PETSc
import dolfinx.fem.petsc
from   dolfinx.fem.petsc import NonlinearProblem, LinearProblem
from   dolfinx.nls.petsc import NewtonSolver
from   mpi4py import MPI


from unit_cell_mpc import get_mpc
from unit_cell_solve import solve_unit_cell
from define_rotation import build_spatial_fields, update_spatial_fields


if not os.path.exists("results"):
    os.makedirs("results")

if not os.path.exists("mesh"):
    os.makedirs("mesh")


##======================================##
##=============== INPUTS ===============##
##======================================##


with open("inputs.yaml", "r") as f:
    cfg = yaml.safe_load(f)

save_every = cfg["save_every"]
num_cycles = cfg["num_cycles"]

temp_ramp_duration       = cfg["temp_ramp_duration"]
temp_long_hold_duration  = cfg["temp_long_hold_duration"]
temp_short_hold_duration = cfg["temp_short_hold_duration"]

initial_temp = cfg["initial_temp"]
final_temp   = cfg["final_temp"]

total_cycle_length = (2 * temp_ramp_duration + temp_long_hold_duration + temp_short_hold_duration)
tf                 = num_cycles * total_cycle_length

num_timesteps = int(tf / 60 / 8)
dt            = tf / num_timesteps

kin      = cfg["pyrolysis"]
n        = kin["n"]
A_factor = kin["A_factor"]
E_a      = kin["E_a"]
R_gas    = kin["R_gas"]

vf           = cfg["volume_fractions"]
vf_cer_0     = vf["vf_cer_0"]
vf_poly_0    = vf["vf_poly_0"]
vf_void_0    = vf["vf_void_0"]
a            = vf["a"]
infiltration_ratio = vf["infiltration_ratio"]


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


@dataclass(frozen=False)
class MaterialConstants:
    k: float
    cp: float
    alpha: float
    rho: float
    E1: float
    E2: float
    E3: float
    nu12: float
    nu13: float
    nu23: float
    _G12: float = None
    _G13: float = None
    _G23: float = None

    def __post_init__(self):
        if self._G12 is None: self._G12 = self.E1 / (2.0 * (1.0 + self.nu12))
        if self._G13 is None: self._G13 = self.E1 / (2.0 * (1.0 + self.nu13))
        if self._G23 is None: self._G23 = self.E2 / (2.0 * (1.0 + self.nu23))

    @property
    def G12(self): return self._G12
    @property
    def G13(self): return self._G13
    @property
    def G23(self): return self._G23

vf_fib_micro = 0.61 # completely hard coded, need to calculate the real value
vf_fib_meso  = vf_fib_micro # 0.70 # completely hard coded, need to calculate the real value

fiber   = MaterialConstants(**cfg["fiber"])
polymer = MaterialConstants(**cfg["polymer"])
ceramic = MaterialConstants(**cfg["ceramic"])

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
        self.E1    = fem.Function(self.S, name="E1")
        self.E2    = fem.Function(self.S, name="E2")
        self.E3    = fem.Function(self.S, name="E3")
        self.nu12  = fem.Function(self.S, name="nu12")
        self.nu13  = fem.Function(self.S, name="nu13")
        self.nu23  = fem.Function(self.S, name="nu23")
        self.G12   = fem.Function(self.S, name="G12")
        self.G13   = fem.Function(self.S, name="G13")
        self.G23   = fem.Function(self.S, name="G23")

        self.vf_cer  = vf_cer_0
        self.vf_void = vf_void_0
        self.vf_poly = vf_poly_0

    def rule_of_mixtures_matrix(self, prop_poly, vf_poly, prop_cer, vf_cer):
        return prop_poly * vf_poly + prop_cer * vf_cer

    def rule_of_mixtures_total(self, prop_matrix, prop_fib):
        vf_matrix = 1.0 - self.vf_fib
        return vf_matrix * prop_matrix + self.vf_fib * prop_fib
    
    def update(self, r_old, temp, vf_poly_0, vf_cer_0, vf_void_0):

        r_step = (A_factor * np.exp(-E_a / (R_gas * temp)) * (1.0 - r_old) ** n) * dt + r_old
        
        if (r_step >= r_old).all():
            if (r_step > 1.0 + 1e-6).any():
                raise ValueError("The ceramization degree cannot exceed 1.0. Consider reducing the timestep.")
            r_new = r_step

        self.vf_cer  = (1 - a) * r_new * vf_poly_0 + vf_cer_0
        self.vf_void = a * r_new * vf_poly_0 + vf_void_0
        self.vf_poly = 1.0 - self.vf_cer - self.vf_void

        k_m      = self.rule_of_mixtures_matrix(self.polymer.k, self.vf_poly, self.ceramic.k, self.vf_cer)
        cp_m     = self.rule_of_mixtures_matrix(self.polymer.cp, self.vf_poly, self.ceramic.cp, self.vf_cer)
        alpha_m  = self.rule_of_mixtures_matrix(self.polymer.alpha, self.vf_poly, self.ceramic.alpha, self.vf_cer)
        rho_m    = self.rule_of_mixtures_matrix(self.polymer.rho, self.vf_poly, self.ceramic.rho, self.vf_cer)
        E1_m      = self.rule_of_mixtures_matrix(self.polymer.E1, self.vf_poly, self.ceramic.E1, self.vf_cer)
        E2_m      = self.rule_of_mixtures_matrix(self.polymer.E2, self.vf_poly, self.ceramic.E2, self.vf_cer)
        E3_m      = self.rule_of_mixtures_matrix(self.polymer.E3, self.vf_poly, self.ceramic.E3, self.vf_cer)
        nu12_m    = self.rule_of_mixtures_matrix(self.polymer.nu12, self.vf_poly, self.ceramic.nu12, self.vf_cer)
        nu13_m    = self.rule_of_mixtures_matrix(self.polymer.nu13, self.vf_poly, self.ceramic.nu13, self.vf_cer)
        nu23_m    = self.rule_of_mixtures_matrix(self.polymer.nu23, self.vf_poly, self.ceramic.nu23, self.vf_cer)

        G12_m     = E1_m / (2.0 * (1.0 + nu12_m))
        G13_m     = E1_m / (2.0 * (1.0 + nu13_m))
        G23_m     = E2_m / (2.0 * (1.0 + nu23_m))

        self.r_old.x.array[:] = r_old
        self.r_new.x.array[:] = r_new
        self.k.x.array[:]     = self.rule_of_mixtures_total(k_m,  self.fiber.k)
        self.cp.x.array[:]    = self.rule_of_mixtures_total(cp_m, self.fiber.cp)
        self.alpha.x.array[:] = self.rule_of_mixtures_total(alpha_m,  self.fiber.alpha)
        self.rho.x.array[:]   = self.rule_of_mixtures_total(rho_m,self.fiber.rho)

        self.E1.x.array[:]    = E1_m
        self.E2.x.array[:]    = E2_m
        self.E3.x.array[:]    = E3_m
        self.nu12.x.array[:]  = nu12_m
        self.nu13.x.array[:]  = nu13_m
        self.nu23.x.array[:]  = nu23_m
        self.G12.x.array[:]   = G12_m
        self.G13.x.array[:]   = G13_m
        self.G23.x.array[:]   = G23_m

        for f in (self.r_old, self.r_new, self.k, self.cp, self.alpha, self.rho,
                self.E1, self.E2, self.E3, self.nu12, self.nu13, self.nu23, self.G12, self.G13, self.G23):
            f.x.scatter_forward()

        return r_new

material_state_micro = MaterialState(domain, fiber, polymer, ceramic, vf_fib_micro)
vf_poly_point_values = [np.mean(material_state_micro.vf_poly)]
vf_cer_point_values = [np.mean(material_state_micro.vf_cer)]
vf_void_point_values = [np.mean(material_state_micro.vf_void)]


##==============================================##
##=== CALCULATE STARTING MATERIAL PROPERTIES ===##
##==============================================##


u_temp_prev.interpolate(lambda x: np.full(x.shape[1], initial_temp, dtype=default_scalar_type)) # is interpolate necessary
delta_temp = u_temp_prev - initial_temp

with XDMFFile(MPI.COMM_WORLD, "mesh/micro_rve_3D.xdmf", "r") as xdmf:
    domain_micro    = xdmf.read_mesh(name="Grid")
    cell_tags_micro = xdmf.read_meshtags(domain_micro, name="Grid")      

material_state_micro.update(material_state_micro.r_new.x.array, u_temp_prev.x.array, vf_poly_0, vf_cer_0, vf_void_0)
mpc_micro, bcs_disp_micro = get_mpc(domain_micro)

beta_history_micro                 = fem.Constant(domain_micro, 0.0)
stiffness_tensor_homogenized_micro = fem.Constant(domain_micro, np.zeros((6, 6), dtype=default_scalar_type))
eigenstrain_homogenized_micro      = fem.Constant(domain_micro, np.zeros(6, dtype=default_scalar_type))

solve_unit_cell('micro', domain_micro, cell_tags_micro, material_state_micro, mpc_micro, bcs_disp_micro, 
                u_temp_prev, beta_history_micro, stiffness_tensor_homogenized_micro, eigenstrain_homogenized_micro, fem.Constant(domain_micro, np.zeros(6, dtype=default_scalar_type)))

def get_mesoscale_properties(stiffness_tensor_homogenized_micro):
    tensor_inv = np.linalg.inv(stiffness_tensor_homogenized_micro)
    E1_tow = 1 / tensor_inv[0,0]
    E2_tow = 1 / tensor_inv[1,1]
    E3_tow = 1 / tensor_inv[2,2]
    nu12_tow = -tensor_inv[0,1] / tensor_inv[0,0]
    nu13_tow = -tensor_inv[0,2] / tensor_inv[0,0]
    nu23_tow = -tensor_inv[1,2] / tensor_inv[1,1]
    G23_tow = 1 / tensor_inv[3,3]
    G13_tow = 1 / tensor_inv[4,4]
    G12_tow = 1 / tensor_inv[5,5]
    return E1_tow, E2_tow, E3_tow, nu12_tow, nu13_tow, nu23_tow, G12_tow, G13_tow, G23_tow

E1_tow, E2_tow, E3_tow, nu12_tow, nu13_tow, nu23_tow, G12_tow, G13_tow, G23_tow = get_mesoscale_properties(stiffness_tensor_homogenized_micro.value)

tow = MaterialConstants(k=np.mean(material_state_micro.k.x.array[:]),
                        cp=np.mean(material_state_micro.cp.x.array[:]),
                        alpha=np.mean(material_state_micro.alpha.x.array[:]),
                        rho=np.mean(material_state_micro.rho.x.array[:]), 
                        E1=E1_tow,
                        E2=E2_tow, 
                        E3=E3_tow, 
                        nu12=nu12_tow, 
                        nu13=nu13_tow, 
                        nu23=nu23_tow, 
                        _G12=G12_tow,
                        _G13=G13_tow,
                        _G23=G23_tow
                       )

with XDMFFile(MPI.COMM_WORLD, "mesh/meso_rve_3D.xdmf", "r") as xdmf:
    domain_meso    = xdmf.read_mesh(name="Grid")
    cell_tags_meso = xdmf.read_meshtags(domain_meso, name="Grid")  

# vals, counts = np.unique(cell_tags_micro.values, return_counts=True)
# print(list(zip(vals, counts)))
# vals, counts = np.unique(cell_tags_meso.values, return_counts=True)
# print(list(zip(vals, counts)))

# print(f'E1: {E1_tow}')
# print(f'E2: {E2_tow}')
# print(f'E3: {E3_tow}')

material_state_meso  = MaterialState(domain, tow, polymer, ceramic, vf_fib_meso)
material_state_meso.update(material_state_meso.r_new.x.array, u_temp_prev.x.array, vf_poly_0, vf_cer_0, vf_void_0)
mpc_meso, bcs_disp_meso = get_mpc(domain_meso)

beta_history_meso = fem.Constant(domain_meso, 0.0)
stiffness_tensor_homogenized_meso = fem.Constant(domain_meso, np.zeros((6, 6), dtype=default_scalar_type))
eigenstrain_homogenized_meso      = fem.Constant(domain_meso, np.zeros(6, dtype=default_scalar_type))
solve_unit_cell('meso', domain_meso, cell_tags_meso, material_state_meso, mpc_meso, bcs_disp_meso,
                u_temp_prev, beta_history_meso, stiffness_tensor_homogenized_meso, eigenstrain_homogenized_meso, eigenstrain_homogenized_micro)

stiffness_spatial, eig_spatial, S_stiffness, S_eig, angle_spatial = build_spatial_fields(
    domain,
    stiffness_tensor_homogenized_meso.value,
    eigenstrain_homogenized_meso.value
)


##======================================##
##======== BOUNDARY CONDITIONS =========##
##======================================##


def is_boundary(x):
    return x[0] <= 1e-12

def all_surfaces(x):
    return np.full(x.shape[1], True, dtype=bool)

fdim_plane = 2
fdim_point = 0

disp_plane         = mesh.locate_entities_boundary(domain, fdim_plane, is_boundary)
disp_vertices      = mesh.locate_entities_boundary(domain, fdim_point, is_boundary)
temp_plane         = mesh.locate_entities_boundary(domain, fdim_plane, all_surfaces)

fixed_dofs_disp_x        = fem.locate_dofs_topological((S_disp.sub(0), S_disp), fdim_plane, disp_plane)
fixed_dofs_disp_vertices = fem.locate_dofs_topological(S_disp, fdim_point, disp_vertices)
fixed_dofs_temp          = fem.locate_dofs_topological(S_temp, fdim_plane, temp_plane)

boundary_temp   = fem.Constant(domain, default_scalar_type(initial_temp))
bcs_temp        = [fem.dirichletbc(boundary_temp, fixed_dofs_temp, S_temp)]

bcs_disp_plane = fem.dirichletbc(
    fem.Constant(domain, default_scalar_type(0.0)), 
    fixed_dofs_disp_x[0],
    S_disp.sub(0)
)
bcs_disp_vertices = fem.dirichletbc(fem.Constant(domain, np.array([0.0, 0.0, 0.0], dtype=default_scalar_type)), fixed_dofs_disp_vertices, S_disp)

bcs_disp = [bcs_disp_plane, bcs_disp_vertices]


##==================================================##
##=== DEFINE THE VARIATIONAL FORM FOR TEMPERATURE===##
##==================================================##


def epsilon(u):
    return ufl.grad(u)

dx = ufl.Measure("dx", domain=domain)

# pretty sure the meso values should be used here (?)
R_temp = material_state_meso.rho * material_state_meso.cp * (u_temp_current - u_temp_prev) / dt * v_temp_current * dx \
        + ufl.dot(material_state_meso.k * epsilon(u_temp_current), epsilon(v_temp_current)) * dx

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

petsc_options={"ksp_type": "preonly", "pc_type": "lu"}

a_disp = ufl.inner(epsilon_sym(v_disp_current), P_tot(u_disp_current, stiffness_spatial)) * dx
L_disp = ufl.inner(epsilon_sym(v_disp_current), P_eigenstrain(eig_spatial, stiffness_spatial)) * dx
problem_disp = LinearProblem(a_disp, L_disp, bcs=bcs_disp, petsc_options=petsc_options)


##=================================================##
##=== DEFINE THE PROJECTION FOR OTHER VARIABLES ===##
##=================================================##


phi    = ufl.TrialFunction(S_constant)
psi    = ufl.TestFunction(S_constant)
a_proj = ufl.inner(phi, psi) * dx

tot_stress_expr = P_tot(u_disp_current_store, stiffness_spatial) - P_eigenstrain(eig_spatial, stiffness_spatial)
von_mises_expr  = ufl.sqrt(0.5 * ((tot_stress_expr[0] - tot_stress_expr[1])**2 +
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

xdmf.write_function(angle_spatial, 0.0)
xdmf.write_function(u_temp_prev, 0.0)
xdmf.write_function(u_disp_current_store, 0.0)
xdmf.write_function(vm_stress_current, 0.0)


##======================================##
##============ TIME STEPPING ===========##
##======================================##


r_avg_values = [np.mean(material_state_micro.r_new.x.array[:])]
r_avg_ref = r_avg_values[0]
temp_avg_values = [np.mean(u_temp_prev.x.array[:])]
vf_poly_avg_values = [np.mean(material_state_micro.vf_poly)]
vf_cer_avg_values = [np.mean(material_state_micro.vf_cer)]
vf_void_avg_values = [np.mean(material_state_micro.vf_void)]

E1_avg_values_meso = [1 / np.linalg.inv(stiffness_tensor_homogenized_meso.value)[0,0]]
E1_point_values_micro = []
E2_point_values_micro = []
E3_point_values_micro = []
G12_point_values_micro = []
G13_point_values_micro = []
G23_point_values_micro = []
transverse_eigenstrain_values_micro = []

E1_point_values_meso = []
E2_point_values_meso = []
E3_point_values_meso = []
G12_point_values_meso = []
G13_point_values_meso = []
G23_point_values_meso = []
transverse_eigenstrain_values_meso = []

time_vector = np.zeros(num_timesteps + 1)

t = dt

old_cycle = 0

pbar = trange(num_timesteps, smoothing=0, colour="green", desc="  Time Stepping", position=0, bar_format='{l_bar}{bar:20}{r_bar}')
for i in pbar:
    pbar.set_postfix({"Cycle": f"{old_cycle + 1}", "Temp": f"{temp_avg_values[-1]:.1f} C", "r": f"{r_avg_values[-1]:.3f}"}, refresh=False)
    
    ##=============================================================================##
    ##=== CALCULATE THE APPLIED TEMPERATURE BASED ON WHERE IN EACH CYCLE WE ARE ===##
    ##=============================================================================##

    current_cycle = np.floor(t / total_cycle_length)

    ramp_up_param = min((t - current_cycle * total_cycle_length) / temp_ramp_duration, 1.0)
    ramp_down_param = max(min((max((t - current_cycle * total_cycle_length), (temp_ramp_duration + temp_long_hold_duration)) \
                        - (temp_ramp_duration + temp_long_hold_duration))  / temp_ramp_duration, 1.0), 0.0)

    temp_bc    = initial_temp + ramp_up_param * (final_temp - initial_temp) - ramp_down_param * (final_temp - initial_temp)
    boundary_temp.value = default_scalar_type(temp_bc)

    ##==========================================##
    ##=== SOLVE AND OUTPUT SOLUTION TO .xdmf ===##
    ##==========================================##

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

    ##=========================================================================================##
    ##=== UPDATE THE HOMOGENIZED PROPERTIES AND "RESET" THE MATERIAL STATE AFTER EACH CYCLE ===##
    ##=========================================================================================##

    if current_cycle != old_cycle:
        vf_poly_0 = infiltration_ratio * material_state_micro.vf_void
        vf_void_0 = (1 - infiltration_ratio) * material_state_micro.vf_void
        vf_cer_0  = material_state_micro.vf_cer
        material_state_micro = MaterialState(domain, fiber, polymer, ceramic, vf_fib_micro)
        material_state_meso  = MaterialState(domain, tow, polymer, ceramic, vf_fib_meso)
        material_state_micro.update(material_state_micro.r_new.x.array, u_temp_prev.x.array, vf_poly_0, vf_cer_0, vf_void_0)
        material_state_meso.update(material_state_meso.r_new.x.array, u_temp_prev.x.array, vf_poly_0, vf_cer_0, vf_void_0)

        E1_point_values_micro.append(1 / np.linalg.inv(stiffness_tensor_homogenized_micro.value)[0,0])
        E2_point_values_micro.append(1 / np.linalg.inv(stiffness_tensor_homogenized_micro.value)[1,1])
        E3_point_values_micro.append(1 / np.linalg.inv(stiffness_tensor_homogenized_micro.value)[2,2])
        G12_point_values_micro.append(1 / np.linalg.inv(stiffness_tensor_homogenized_micro.value)[5,5])
        G13_point_values_micro.append(1 / np.linalg.inv(stiffness_tensor_homogenized_micro.value)[4,4])
        G23_point_values_micro.append(1 / np.linalg.inv(stiffness_tensor_homogenized_micro.value)[3,3])
        transverse_eigenstrain_values_micro.append(eigenstrain_homogenized_micro.value[2])

        E1_point_values_meso.append(1 / np.linalg.inv(stiffness_tensor_homogenized_meso.value)[0,0])
        E2_point_values_meso.append(1 / np.linalg.inv(stiffness_tensor_homogenized_meso.value)[1,1])
        E3_point_values_meso.append(1 / np.linalg.inv(stiffness_tensor_homogenized_meso.value)[2,2])
        G12_point_values_meso.append(1 / np.linalg.inv(stiffness_tensor_homogenized_meso.value)[5,5])
        G13_point_values_meso.append(1 / np.linalg.inv(stiffness_tensor_homogenized_meso.value)[4,4])
        G23_point_values_meso.append(1 / np.linalg.inv(stiffness_tensor_homogenized_meso.value)[3,3])
        transverse_eigenstrain_values_meso.append(eigenstrain_homogenized_meso.value[2])

        vf_poly_point_values.append(np.mean(material_state_micro.vf_poly))
        vf_cer_point_values.append(np.mean(material_state_micro.vf_cer))
        vf_void_point_values.append(np.mean(material_state_micro.vf_void))
        old_cycle = int(current_cycle)
    else:
        material_state_micro.update(material_state_micro.r_new.x.array, u_temp_prev.x.array, vf_poly_0, vf_cer_0, vf_void_0)
        material_state_meso.update(material_state_meso.r_new.x.array, u_temp_prev.x.array, vf_poly_0, vf_cer_0, vf_void_0)
    
    r_avg = np.mean(material_state_micro.r_new.x.array[:])
    
    if abs(r_avg - r_avg_ref) > 1e-3:
        r_avg_ref = r_avg
        solve_unit_cell('micro', domain_micro, cell_tags_micro, material_state_micro, mpc_micro, bcs_disp_micro, 
                        u_temp_prev, beta_history_micro, stiffness_tensor_homogenized_micro, eigenstrain_homogenized_micro, fem.Constant(domain_micro, np.zeros(6, dtype=default_scalar_type)))
        E1_tow, E2_tow, E3_tow, nu12_tow, nu13_tow, nu23_tow, G12_tow, G13_tow, G23_tow = get_mesoscale_properties(stiffness_tensor_homogenized_micro.value)
        tow = MaterialConstants(k=np.mean(material_state_micro.k.x.array[:]),
                                cp=np.mean(material_state_micro.cp.x.array[:]),
                                alpha=np.mean(material_state_micro.alpha.x.array[:]),
                                rho=np.mean(material_state_micro.rho.x.array[:]), 
                                E1=E1_tow,
                                E2=E2_tow,
                                E3=E3_tow,  
                                nu12=nu12_tow,
                                nu13=nu13_tow,
                                nu23=nu23_tow, 
                                _G12=G12_tow,
                                _G13=G13_tow,
                                _G23=G23_tow
                            )
        material_state_meso.fiber = tow
        solve_unit_cell('meso', domain_meso, cell_tags_meso, material_state_meso, mpc_meso, bcs_disp_meso,
                        u_temp_prev, beta_history_meso, stiffness_tensor_homogenized_meso, eigenstrain_homogenized_meso, eigenstrain_homogenized_micro)
        
        update_spatial_fields(stiffness_spatial, eig_spatial, domain,
                              stiffness_tensor_homogenized_meso.value,
                              eigenstrain_homogenized_meso.value)
        
    ##==============================================##
    ##=== CALCULATE MEAN QUANTITIES FOR PLOTTING ===##
    ##==============================================##

    time_vector[i + 1] = t / 60 / 60
    
    temp_avg = np.mean(u_temp_prev.x.array[:])
    vf_poly_avg = np.mean(material_state_micro.vf_poly)
    vf_cer_avg = np.mean(material_state_micro.vf_cer)
    vf_void_avg = np.mean(material_state_micro.vf_void)

    r_avg_values.append(r_avg)
    temp_avg_values.append(temp_avg)
    vf_poly_avg_values.append(vf_poly_avg)
    vf_cer_avg_values.append(vf_cer_avg)
    vf_void_avg_values.append(vf_void_avg)
    E1_avg_values_meso.append(1 / np.linalg.inv(stiffness_tensor_homogenized_meso.value)[0,0])

    t += dt

xdmf.close()


##===========================================##
##=== PLOT QUANTITIES AT THE CENTER POINT ===##
##===========================================##


marker_size = 6
axis_font_size = 18
legend_font_size = 14
ramp_up_end_index = int(temp_ramp_duration / dt)

plt.figure(1)
plt.plot(time_vector, temp_avg_values, marker='o', color='r')
plt.xlabel('Time (hr)', fontsize=axis_font_size)
plt.ylabel('Temperature (C)', fontsize=axis_font_size)
plt.grid(True)
plt.savefig('results/temp_vs_time.png')

plt.figure(2)
plt.plot(time_vector, vf_poly_avg_values, color='green', linewidth = 3, label='Precursor')
plt.plot(time_vector, vf_cer_avg_values, color='orange', linewidth = 3, label='Ceramic')
plt.plot(time_vector, vf_void_avg_values, color='gray', linewidth = 3, label='Voids')
plt.xlabel('Time (hr)', fontsize=axis_font_size)
plt.ylabel('Volume Fraction (%)', fontsize=axis_font_size)
plt.legend(fontsize=legend_font_size)
plt.grid(True)
plt.ylim([0,1])
plt.savefig('results/volume_fractions_vs_time.png')

plt.figure(3)
ax1 = plt.gca()
ax2 = ax1.twinx()
ax2.plot(time_vector, temp_avg_values, '--r', linewidth=3, label='Temperature')
ax1.plot(time_vector, E1_avg_values_meso, '-b', linewidth=3, label='Elastic Modulus')
ax1.set_xlabel('Time (hr)', fontsize=axis_font_size)
ax1.set_ylabel('Axial Elastic Modulus, E1 (Pa)', fontsize=axis_font_size)
ax2.set_ylabel('Temperature (C)', fontsize=axis_font_size)
plt.grid(True)
plt.savefig('results/E_and_temp_vs_time.png')

plt.figure(4)
plt.plot(temp_avg_values[0:ramp_up_end_index], r_avg_values[0:ramp_up_end_index], marker='o', color='orange')
plt.xlabel('Temperature (C)', fontsize=axis_font_size)
plt.ylabel('r', fontsize=axis_font_size)
plt.grid(True)
plt.savefig('results/r_vs_temp.png')

plt.figure(5)
cycle_num = np.arange(1, num_cycles + 1)
plt.plot(cycle_num, E1_point_values_micro, marker='s', markersize=marker_size, color='gray', label=r'$E_1$')
plt.plot(cycle_num, E2_point_values_micro, marker='o', markersize=marker_size, color='red', label=r'$E_2$')
plt.plot(cycle_num, E3_point_values_micro, marker='o', markersize=marker_size, color='red', label=r'$E_3$')
plt.plot(cycle_num, G12_point_values_micro, marker='^', markersize=marker_size, color='blue', label=r'$G_{12}$')
plt.plot(cycle_num, G13_point_values_micro, marker='^', markersize=marker_size, color='blue', label=r'$G_{13}$')
plt.plot(cycle_num, G23_point_values_micro, marker='v', markersize=marker_size, color='green', label=r'$G_{23}$')
plt.xlabel('Cycle Number', fontsize=axis_font_size)
plt.ylabel('Modulus (GPa)', fontsize=axis_font_size)
plt.legend(fontsize=legend_font_size)
plt.xticks(cycle_num)
plt.xlim([1, num_cycles])
plt.yticks(np.arange(20e9, 230e9 + 1, 20e9))
plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1e9:.0f}'))
plt.grid(True)
plt.savefig('results/elastic_properties_vs_cycle_micro.png')

plt.figure(6)
cycle_num = np.arange(1, num_cycles + 1)
plt.plot(cycle_num, E1_point_values_meso, marker='s', markersize=marker_size, color='gray', label=r'$E_1$')
plt.plot(cycle_num, E2_point_values_meso, marker='s', markersize=marker_size, color='gray', label=r'$E_2$')
plt.plot(cycle_num, E3_point_values_meso, marker='o', markersize=marker_size, color='red', label=r'$E_3$')
plt.plot(cycle_num, G12_point_values_meso, marker='^', markersize=marker_size, color='blue', label=r'$G_{12}$')
plt.plot(cycle_num, G13_point_values_meso, marker='v', markersize=marker_size, color='green', label=r'$G_{13}$')
plt.plot(cycle_num, G23_point_values_meso, marker='v', markersize=marker_size, color='green', label=r'$G_{23}$')
plt.xlabel('Cycle Number', fontsize=axis_font_size)
plt.ylabel('Modulus (GPa)', fontsize=axis_font_size)
plt.legend(fontsize=legend_font_size)
plt.xticks(cycle_num)
plt.xlim([1, num_cycles])
plt.yticks(np.arange(20e9, 230e9 + 1, 20e9))
plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1e9:.0f}'))
plt.grid(True)
plt.savefig('results/elastic_properties_vs_cycle_meso.png')

plt.figure(7)
plt.plot(cycle_num, transverse_eigenstrain_values_micro, marker='s', markersize=marker_size, color='gray', label=r'$\varepsilon_2 = \varepsilon_3$')
plt.xlabel('Cycle Number', fontsize=axis_font_size)
plt.ylabel(r'$\varepsilon$', fontsize=axis_font_size)
plt.xticks(cycle_num)
plt.xlim([1, num_cycles])
plt.legend(fontsize=legend_font_size)
plt.grid(True)
plt.savefig('results/transverse_eigenstrain_vs_cycle_micro.png')

plt.figure(8)
plt.plot(cycle_num, transverse_eigenstrain_values_meso, marker='s', markersize=marker_size, color='gray', label=r'$\varepsilon_3$')
plt.xlabel('Cycle Number', fontsize=axis_font_size)
plt.ylabel(r'$\varepsilon$', fontsize=axis_font_size)
plt.xticks(cycle_num)
plt.xlim([1, num_cycles])
plt.legend(fontsize=legend_font_size)
plt.grid(True)
plt.savefig('results/transverse_eigenstrain_vs_cycle_meso.png')

plt.figure(9)
cycle_num = np.arange(0, num_cycles + 1)
plt.plot(cycle_num, vf_poly_point_values, marker='s', markersize=marker_size, color='gray', label='Polymer Volume Fraction')
plt.plot(cycle_num, vf_void_point_values, marker='o', markersize=marker_size, color='red', label='Void Volume Fraction')
plt.plot(cycle_num, vf_cer_point_values, marker='^', markersize=marker_size, color='blue', label='Ceramic Volume Fraction')
plt.xlabel('Cycle Number', fontsize=axis_font_size)
plt.ylabel('Volume Fraction (%)', fontsize=axis_font_size)
plt.xticks(cycle_num)
plt.xlim([0, num_cycles])
plt.ylim([0,1])
plt.legend(fontsize=legend_font_size)
plt.grid(True)
plt.savefig('results/volume_fractions_vs_cycle.png')