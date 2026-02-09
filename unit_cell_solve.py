import numpy as np
import matplotlib.pyplot as plt
import os
from   tqdm import tqdm
from   tqdm.auto import trange

import ufl
import dolfinx_mpc.utils
from   dolfinx import fem
from   dolfinx_mpc import LinearProblem as MPCLinearProblem
from   mpi4py import MPI


def solve_unit_cell(domain, cell_tags, material_state, mpc, bcs_disp, u_temp_prev, stiffness_tensor_homogenized, eigenstrain_homogenized):

    length   = 24e-6
    width    = 24e-6
    height   = 24e-6    

    unit_cell_volume = length * width * height

    S = fem.functionspace(domain, ("Lagrange", 1, (domain.geometry.dim, )))
    
    h = ufl.TrialFunction(S)
    h_ = ufl.TestFunction(S)
    k  = ufl.TrialFunction(S)
    k_ = ufl.TestFunction(S)

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
    
    dim = domain.geometry.dim
    I = ufl.variable(ufl.Identity(dim))

    def epsilon_volume(r):
        beta = -0.3 * r ** 2 + 0.22 * r - 0.17
        return get_voigt(beta * r * I)

    def epsilon_thermal(alpha, delta_temp):
        return get_voigt(alpha * delta_temp * I)

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

    def P_tot_multiple_rhs(h, stiffness_matrix):
        epsilon_tot = applied_eps + epsilon_sym(h)
        sigma_voigt = ufl.dot(stiffness_matrix, epsilon_tot)
        return sigma_voigt

    def P_tot(k, stiffness_matrix):
        return ufl.dot(stiffness_matrix, epsilon_sym(k))

    mu_const     = fem.Constant(domain, np.mean(material_state.mu.x.array[:]))
    lam_const    = fem.Constant(domain, np.mean(material_state.lam.x.array[:]))
    alpha_const  = fem.Constant(domain, np.mean(material_state.alpha.x.array[:]))
    r_const      = fem.Constant(domain, np.mean(material_state.r.x.array[:]))
    
    material_properties = {
        1: (material_state.fiber.mu, material_state.fiber.lam, material_state.fiber.alpha),
        2: (mu_const, lam_const, alpha_const)
    }

    dx = ufl.Measure("dx", domain=domain, subdomain_data=cell_tags)

    a_h = 0.0
    L_h = 0.0
    a_k = 0.0
    L_k = 0.0

    delta_temp_value = u_temp_prev.x.array[:].mean() - 0.0 # hard coded
    delta_temp = fem.Constant(domain, delta_temp_value)

    for tag, (mu, lam, alpha) in material_properties.items():
        stiffness_matrix = get_stiffness_tensor(mu, lam)
        a_h_tag, L_h_tag = ufl.system(ufl.inner(P_tot_multiple_rhs(h, stiffness_matrix), epsilon_sym(h_)) * dx(tag))
        a_h += a_h_tag
        L_h += L_h_tag

        if tag == 2:
            a_k += ufl.inner(epsilon_sym(k_), P_tot(k, stiffness_matrix)) * dx(tag)
            L_k += ufl.inner(epsilon_sym(k_), ufl.dot(stiffness_matrix, epsilon_thermal(alpha, delta_temp) + epsilon_volume(r_const))) * dx(tag)
        else:    
            a_k += ufl.inner(epsilon_sym(k_), P_tot(k, stiffness_matrix)) * dx(tag)
            L_k += ufl.inner(epsilon_sym(k_), ufl.dot(stiffness_matrix, epsilon_thermal(alpha, delta_temp))) * dx(tag)       

    u_mpc_h = fem.Function(mpc.function_space)
    u_mpc_k = fem.Function(mpc.function_space)

    problem_h = MPCLinearProblem(a_h, L_h, mpc, bcs=[bcs_disp], u=u_mpc_h)
    problem_k = MPCLinearProblem(a_k, L_k, mpc, bcs=[bcs_disp], u=u_mpc_k)
    K_solve   = problem_k.solve()
    
    elementary_load = np.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0], 
                                [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                                [0.0, 0.0, 1.0, 0.0, 0.0, 0.0], 
                                [0.0, 0.0, 0.0, 1.0, 0.0, 0.0], 
                                [0.0, 0.0, 0.0, 0.0, 1.0, 0.0], 
                                [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])

    dim_load = elementary_load.shape[0]
    temporary_tensor = np.zeros((dim_load, dim_load))

    for i in trange(dim_load, colour="red", desc="Solve Unit Cell", position=1, leave=False, bar_format='{l_bar}{bar:30}{r_bar}', total=dim_load):
        applied_eps.value = elementary_load[i]
        H_solve = problem_h.solve()
        
        for j in range(dim_load):
            if ((i > 2) | (j > 2)) & (i != j):
                continue
            applied_eps_.value = elementary_load[j]
            for tag, (mu, lam, _) in material_properties.items():
                stiffness_matrix = get_stiffness_tensor(mu, lam)
                temporary_tensor[i, j] += (1 / unit_cell_volume) * fem.assemble_scalar(fem.form(ufl.inner(P_tot_multiple_rhs(H_solve, stiffness_matrix), applied_eps_) * dx(tag)))

    stiffness_tensor_homogenized.value = temporary_tensor

    temporary_vector = np.zeros((dim_load))
    for tag, (mu, lam, alpha) in material_properties.items():
        stiffness_matrix = get_stiffness_tensor(mu, lam)
        for j in range(dim_load):
            if j > 2:
                continue
            applied_eps_.value = elementary_load[j]
            if tag == 2:
                temporary_vector[j] += fem.assemble_scalar(fem.form(ufl.inner(ufl.dot(stiffness_matrix, epsilon_sym(K_solve)), applied_eps_) * dx(tag)))
                temporary_vector[j] -= fem.assemble_scalar(fem.form(ufl.inner(ufl.dot(stiffness_matrix, epsilon_thermal(alpha, delta_temp) + epsilon_volume(r_const)), applied_eps_) * dx(tag)))
            else:
                temporary_vector[j] += fem.assemble_scalar(fem.form(ufl.inner(ufl.dot(stiffness_matrix, epsilon_sym(K_solve)), applied_eps_) * dx(tag)))
                temporary_vector[j] -= fem.assemble_scalar(fem.form(ufl.inner(ufl.dot(stiffness_matrix, epsilon_thermal(alpha, delta_temp)), applied_eps_) * dx(tag)))
                
    mu_bar = -(1 / unit_cell_volume) * np.linalg.inv(temporary_tensor) @ temporary_vector
    eigenstrain_homogenized.value = mu_bar