import numpy as np
import matplotlib.pyplot as plt
import os
from   tqdm import tqdm
from   tqdm.auto import trange

import ufl
import dolfinx_mpc.utils
from   dolfinx import fem, default_scalar_type
from   dolfinx_mpc import LinearProblem as MPCLinearProblem
from   mpi4py import MPI


def solve_unit_cell(domain, cell_tags, material_state, mpc, bcs_disp, u_temp_prev, beta_history, stiffness_tensor_homogenized, eigenstrain_homogenized):

    length   = 16e-6
    width    = 16e-6
    height   = 16e-6    

    unit_cell_volume = length * width * height

    E1_avg       = np.mean(material_state.E1.x.array[:])
    E2_avg       = np.mean(material_state.E2.x.array[:])
    E3_avg       = np.mean(material_state.E3.x.array[:])
    nu12_avg     = np.mean(material_state.nu12.x.array[:])
    nu13_avg     = np.mean(material_state.nu13.x.array[:])
    nu23_avg     = np.mean(material_state.nu23.x.array[:])
    G12_avg      = np.mean(material_state.G12.x.array[:])
    G13_avg      = np.mean(material_state.G13.x.array[:])
    G23_avg      = np.mean(material_state.G23.x.array[:])
    alpha_avg    = fem.Constant(domain, np.mean(material_state.alpha.x.array[:]))
    vf_poly_avg  = fem.Constant(domain, np.mean(material_state.vf_poly))
    r_new_avg    = fem.Constant(domain, np.mean(material_state.r_new.x.array[:]))
    r_old_avg    = fem.Constant(domain, np.mean(material_state.r_old.x.array[:]))

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

    def get_beta(r_new, r_old):
        return vf_poly_avg * ((-0.3 * r_new ** 3 + 0.22 * r_new ** 2 - 0.17 * r_new) \
                              - (-0.3 * r_old ** 3 + 0.22 * r_old ** 2 - 0.17 * r_old))

    def epsilon_volume(beta_current, beta_history):
        return get_voigt((beta_current + beta_history) * I)

    def epsilon_thermal(alpha, delta_temp):
        return get_voigt(alpha * delta_temp * I)

    def get_stiffness_tensor(E1, E2, E3, nu12, nu13, nu23, G12, G13, G23):
        S = np.array([[1 / E1, -nu12 / E1, -nu13 / E1, 0, 0, 0],
                           [-nu12 / E1, 1 / E2, -nu23 / E2, 0, 0, 0],
                           [-nu13 / E1, -nu23 / E2, 1 / E3, 0, 0, 0],
                           [0, 0, 0, 1 / G23, 0, 0],
                           [0, 0, 0, 0, 1 / G13, 0],
                           [0, 0, 0, 0, 0, 1 / G12]])
        C = np.linalg.inv(S)
        C_ufl = fem.Constant(domain, C)
        return C_ufl
    
    applied_eps  = fem.Constant(domain, np.zeros((6)))
    applied_eps_ = fem.Constant(domain, np.zeros((6)))

    def P_tot_multiple_rhs(h, stiffness_matrix):
        epsilon_tot = applied_eps + epsilon_sym(h)
        sigma_voigt = ufl.dot(stiffness_matrix, epsilon_tot)
        return sigma_voigt

    def P_tot(k, stiffness_matrix):
        return ufl.dot(stiffness_matrix, epsilon_sym(k))

    material_properties = {
        1: (material_state.fiber.E1, material_state.fiber.E2, material_state.fiber.E3, material_state.fiber.nu12, material_state.fiber.nu13, material_state.fiber.nu23, material_state.fiber.G12, material_state.fiber.G13, material_state.fiber.G23, material_state.fiber.alpha),
        2: (E1_avg, E2_avg, E3_avg, nu12_avg, nu13_avg, nu23_avg, G12_avg, G13_avg, G23_avg, alpha_avg)
    }

    stiffness_matrices = {}
    for tag, props in material_properties.items():
        E1_p, E2_p, E3_p, nu12_p, nu13_p, nu23_p, G12_p, G13_p, G23_p, _alpha_p = props
        stiffness_matrices[tag] = get_stiffness_tensor(E1_p, E2_p, E3_p, nu12_p, nu13_p, nu23_p, G12_p, G13_p, G23_p)


    dx = ufl.Measure("dx", domain=domain, subdomain_data=cell_tags)

    a_h = 0.0
    L_h = 0.0
    a_k = 0.0
    L_k = 0.0

    delta_temp_value = u_temp_prev.x.array[:].mean() - 200.0 # hard coded (check later !)
    delta_temp = fem.Constant(domain, delta_temp_value)
    beta_current = get_beta(r_new_avg, r_old_avg)

    for tag, (E1, E2, E3, nu12, nu13, nu23, G12, G13, G23, alpha) in material_properties.items():
        stiffness_matrix = stiffness_matrices[tag]
        a_h_tag, L_h_tag = ufl.system(ufl.inner(P_tot_multiple_rhs(h, stiffness_matrix), epsilon_sym(h_)) * dx(tag))
        a_h += a_h_tag
        L_h += L_h_tag

        if tag == 2:
            a_k += ufl.inner(epsilon_sym(k_), P_tot(k, stiffness_matrix)) * dx(tag)
            L_k += ufl.inner(epsilon_sym(k_), ufl.dot(stiffness_matrix, epsilon_thermal(alpha, delta_temp) + epsilon_volume(beta_current, beta_history))) * dx(tag)
        else:    
            a_k += ufl.inner(epsilon_sym(k_), P_tot(k, stiffness_matrix)) * dx(tag)
            L_k += ufl.inner(epsilon_sym(k_), ufl.dot(stiffness_matrix, epsilon_thermal(alpha, delta_temp))) * dx(tag)       

    u_mpc_h = fem.Function(mpc.function_space)
    u_mpc_k = fem.Function(mpc.function_space)

    petsc_options={}

    problem_h = MPCLinearProblem(a_h, L_h, mpc, bcs=[bcs_disp], u=u_mpc_h, petsc_options=petsc_options)
    problem_k = MPCLinearProblem(a_k, L_k, mpc, bcs=[bcs_disp], u=u_mpc_k, petsc_options=petsc_options)
    K_solve   = problem_k.solve()
    
    elementary_load = np.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0], 
                                [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                                [0.0, 0.0, 1.0, 0.0, 0.0, 0.0], 
                                [0.0, 0.0, 0.0, 1.0, 0.0, 0.0], 
                                [0.0, 0.0, 0.0, 0.0, 1.0, 0.0], 
                                [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])

    dim_load = elementary_load.shape[0]
    temporary_tensor = np.zeros((dim_load, dim_load))
    vol_inv = 1.0 / unit_cell_volume
    j_allowed = {i: [j for j in range(dim_load) if not (((i > 2) or (j > 2)) and (i != j))] for i in range(dim_load)}

    # for i in trange(dim_load, colour="red", desc="Solve Unit Cell", position=1, leave=False, bar_format='{l_bar}{bar:30}{r_bar}', total=dim_load):
    for i in range(dim_load):
        applied_eps.value = elementary_load[i]
        H_solve = problem_h.solve()

        for j in j_allowed[i]:
            applied_eps_.value = elementary_load[j]
            for tag, stiffness_matrix in stiffness_matrices.items():
                temporary_tensor[i, j] += vol_inv * fem.assemble_scalar(fem.form(ufl.inner(P_tot_multiple_rhs(H_solve, stiffness_matrix), applied_eps_) * dx(tag)))

    stiffness_tensor_homogenized.value = temporary_tensor

    temporary_vector = np.zeros((dim_load))
    for tag, (E1, E2, E3, nu12, nu13, nu23, G12, G13, G23, alpha) in material_properties.items():
        stiffness_matrix = stiffness_matrices[tag]
        for j in range(dim_load):
            if j > 2:
                continue
            applied_eps_.value = elementary_load[j]
            if tag == 2:
                temporary_vector[j] += fem.assemble_scalar(fem.form(ufl.inner(ufl.dot(stiffness_matrix, epsilon_sym(K_solve)), applied_eps_) * dx(tag)))
                temporary_vector[j] -= fem.assemble_scalar(fem.form(ufl.inner(ufl.dot(stiffness_matrix, epsilon_thermal(alpha, delta_temp) + epsilon_volume(beta_current, beta_history)), applied_eps_) * dx(tag)))
            else:
                temporary_vector[j] += fem.assemble_scalar(fem.form(ufl.inner(ufl.dot(stiffness_matrix, epsilon_sym(K_solve)), applied_eps_) * dx(tag)))
                temporary_vector[j] -= fem.assemble_scalar(fem.form(ufl.inner(ufl.dot(stiffness_matrix, epsilon_thermal(alpha, delta_temp)), applied_eps_) * dx(tag)))
                
    eigenstrain_bar = -(1 / unit_cell_volume) * np.linalg.inv(temporary_tensor) @ temporary_vector
    eigenstrain_homogenized.value = eigenstrain_bar

    beta_history.value += float(beta_current)