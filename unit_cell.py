import numpy as np
import matplotlib.pyplot as plt
import os
from   tqdm import tqdm
from   tqdm.auto import trange

import ufl
import dolfinx_mpc.utils
from   dolfinx import fem, mesh, plot, default_scalar_type
from   dolfinx.io import XDMFFile
from   dolfinx_mpc import LinearProblem as MPCLinearProblem
from   mpi4py import MPI


def solve_unit_cell(domain, cell_tags, material_state, mpc, bcs_disp):

    length   = 24e-6
    width    = 24e-6
    height   = 24e-6    

    S_disp = fem.functionspace(domain, ("Lagrange", 1, (domain.geometry.dim, )))

    h = ufl.TrialFunction(S_disp)
    k = ufl.TestFunction(S_disp)

    def get_voigt(tensor):
        return ufl.as_vector([   tensor[0, 0],   # ₁₁
                                 tensor[1, 1],   # ₂₂
                                 tensor[2, 2],   # ₃₃
                             2 * tensor[1, 2],   # ₂₃ 
                             2 * tensor[0, 2],   # ₁₃ 
                             2 * tensor[0, 1]    # ₁₂ 
                            ])

    def epsilon_sym(u):
        epsilon = ufl.sym(ufl.grad(u))
        return get_voigt(epsilon)

    def get_stiffness_tensor(mu, lam):
        C = ufl.as_matrix([[lam + 2 * mu,     lam,           lam,           0,              0,              0],
                           [lam,           lam + 2 * mu,     lam,           0,              0,              0],
                           [lam,               lam,       lam + 2 * mu,     0,              0,              0],
                           [0,                 0,              0,          mu,              0,              0],
                           [0,                 0,              0,           0,             mu,              0],
                           [0,                 0,              0,           0,              0,             mu]])
        return C

    applied_eps  = fem.Constant(domain, np.zeros((6)))
    applied_eps_ = fem.Constant(domain, np.zeros((6)))

    def P_tot(h, stiffness_matrix):
        epsilon_tot = applied_eps + epsilon_sym(h)
        sigma_voigt = ufl.dot(stiffness_matrix, epsilon_tot)
        return sigma_voigt

    material_properties = {
        1: (material_state.fiber.mu, material_state.fiber.lam, material_state.fiber.alpha),
        2: (material_state.mu, material_state.lam, material_state.alpha)
    }

    dx = ufl.Measure("dx", domain=domain, subdomain_data=cell_tags)
    a_disp = 0.0

    for tag, (mu, lam, _) in material_properties.items():
        stiffness_matrix = get_stiffness_tensor(mu, lam)
        a_tag, L_disp = ufl.system(ufl.inner(P_tot(h, stiffness_matrix), epsilon_sym(k)) * dx(tag))
        a_disp += a_tag

    u_mpc = fem.Function(mpc.function_space)
    problem_disp = MPCLinearProblem(a_disp, L_disp, mpc, bcs=[bcs_disp], u=u_mpc)
    
    elementary_load = np.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0], 
                                [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                                [0.0, 0.0, 1.0, 0.0, 0.0, 0.0], 
                                [0.0, 0.0, 0.0, 1.0, 0.0, 0.0], 
                                [0.0, 0.0, 0.0, 0.0, 1.0, 0.0], 
                                [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])

    dim_load = elementary_load.shape[0]
    D_homogenized_matrix = np.ones((dim_load, dim_load))
    
    unit_cell_volume = length * width * height

    for i in trange(dim_load, colour="red", desc="Solve Unit Cell", position=1, leave=False, bar_format='{l_bar}{bar:30}{r_bar}', total=dim_load):
        applied_eps.value = elementary_load[i]
        h_solve = problem_disp.solve()
        
        for j in range(dim_load):
            applied_eps_.value = elementary_load[j]
            for tag, (mu, lam, _) in material_properties.items():
                stiffness_matrix = get_stiffness_tensor(mu, lam)
                D_homogenized_matrix[i, j] += (1 / unit_cell_volume) * fem.assemble_scalar(fem.form(ufl.inner(P_tot(h_solve, stiffness_matrix), applied_eps_) * dx(tag)))

    D_homogenized_matrix = ufl.as_matrix(D_homogenized_matrix)

    return D_homogenized_matrix