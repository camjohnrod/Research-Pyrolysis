import os
from dolfinx import fem, mesh, io, plot, default_scalar_type
from dolfinx.io import XDMFFile
import ufl
from dolfinx_mpc import LinearProblem as MPCLinearProblem
from dolfinx.fem.petsc import NonlinearProblem, LinearProblem
from dolfinx.nls.petsc import NewtonSolver
import dolfinx_mpc.utils
from mpi4py import MPI
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

def solve_unit_cell(r_func):

    ## Inputs ##

    # t0               = 0.0
    # tf               = 10800
    # ramp_duration    = 10800
    # num_steps        = int(1)
    # dt               = tf / num_steps

    E_polymer        = 4.94e9    # updated
    E_ceramic        = 206.18e9  # updated
    E_fiber          = 264.5e9

    nu_polymer       = 0.3
    nu_ceramic       = 0.14
    nu_fiber         = 0.14

    alpha_matrix     = 3.95e-6
    alpha_fiber      = -0.64e-6

    initial_temp     = 0.0
    final_temp       = 1200.0

    length           = 24e-6
    width            = 24e-6
    height           = 24e-6

    vf_ceramic_0     = 0.0
    vf_polymer_0     = 1.0
    a                = 0.05

    n                = 2
    A_factor         = 1.5795 / (60)
    E_a              = 18216.0
    R_gas            = 8.3145

    ## Define mesh ##

    with XDMFFile(MPI.COMM_WORLD, "mesh/rve_3D.xdmf", "r") as xdmf:
        domain = xdmf.read_mesh(name="Grid")
        cell_tags = xdmf.read_meshtags(domain, name="Grid")      


    ## Define material properties ##

    S_mat_prop = fem.functionspace(domain, ("Lagrange", 1))

    alpha_func = fem.Function(S_mat_prop, name="alpha")
    # r_func = fem.Function(S_mat_prop, name="r")
    E_func = fem.Function(S_mat_prop, name="E")
    nu_func = fem.Function(S_mat_prop, name="nu")
    mu_func = fem.Function(S_mat_prop, name="mu")
    lam_func = fem.Function(S_mat_prop, name="lam")

    def get_vf_ceramic(r, a, vf_polymer_0):
        return (1 - a) * r * vf_polymer_0

    def get_vf_void(r, a, vf_polymer_0):
        return a * r * vf_polymer_0

    def get_vf_polymer(vf_ceramic, vf_ceramic_0, vf_void):
        return 1 - vf_ceramic - vf_ceramic_0 - vf_void 

    def rule_of_mixtures(prop_polymer, vf_polymer, prop_ceramic, vf_ceramic):
        return prop_polymer * vf_polymer + prop_ceramic * vf_ceramic

    def update_matrix_material_properties():

        # r_new = A_factor * (np.exp(-E_a / (R_gas * (u_temp_prev + 273.15)))) * (1 - r_old) ** n * dt + r_old

        vf_ceramic = get_vf_ceramic(r_func, a, vf_polymer_0)
        vf_void = get_vf_void(r_func, a, vf_polymer_0)
        vf_polymer = get_vf_polymer(vf_ceramic, vf_ceramic_0, vf_void)

        alpha_val = rule_of_mixtures(alpha_matrix, vf_polymer, alpha_matrix, vf_ceramic)
        E_val = rule_of_mixtures(E_polymer, vf_polymer, E_ceramic, vf_ceramic)
        nu_val = rule_of_mixtures(nu_polymer, vf_polymer, nu_ceramic, vf_ceramic)
        mu_val = E_val / (2 * (1 + nu_val))
        lam_val = E_val * nu_val / ((1 + nu_val) * (1 - 2 * nu_val))
        
        alpha_func.x.array[:] = alpha_val
        # r_func.x.array[:] = r_new
        E_func.x.array[:] = E_val
        nu_func.x.array[:] = nu_val
        mu_func.x.array[:] = mu_val
        lam_func.x.array[:] = lam_val

        alpha_func.x.scatter_forward()
        # r_func.x.scatter_forward()
        E_func.x.scatter_forward()
        nu_func.x.scatter_forward()
        mu_func.x.scatter_forward()
        lam_func.x.scatter_forward()

    mu_fiber = E_fiber / (2 * (1 + nu_fiber))
    lam_fiber = E_fiber * nu_fiber / ((1 + nu_fiber) * (1 - 2 * nu_fiber))

    material_properties = { 
        1: (mu_fiber, lam_fiber, alpha_fiber),
        2: (mu_func, lam_func, alpha_func)
    }


    ## Define temperature function spaces ##

    # S_temp = fem.functionspace(domain, ("Lagrange", 1))

    # u_temp_prev = fem.Function(S_temp)
    # u_temp_prev.name = "Temperature"


    ## Define each boundary (faces, edges, and vertices) ##

    def vertices(x):
        return np.isclose(x[0], 0) & (np.isclose(x[1], 0) | np.isclose(x[1], width)) & \
                (np.isclose(x[2], 0) | np.isclose(x[2], height)) | \
                np.isclose(x[0], length) & (np.isclose(x[1], 0) | np.isclose(x[1], width)) & \
                (np.isclose(x[2], 0) | np.isclose(x[2], height))  

    def left(x):
        return (np.isclose(x[0], 0) & (x[1] > 0) & (x[1] < width) & (x[2] > 0) & (x[2] < height))    
    def right(x):
        return (np.isclose(x[0], length) & (x[1] > 0) & (x[1] < width) & (x[2] > 0) & (x[2] < height))
    def front(x):
        return (np.isclose(x[1], 0) & (x[0] > 0) & (x[0] < length) & (x[2] > 0) & (x[2] < height))
    def back(x):
        return (np.isclose(x[1], width) & (x[0] > 0) & (x[0] < length) & (x[2] > 0) & (x[2] < height))
    def bottom(x):
        return (np.isclose(x[2], 0) & (x[0] > 0) & (x[0] < length) & (x[1] > 0) & (x[1] < width))
    def top(x):
        return (np.isclose(x[2], height) & (x[0] > 0) & (x[0] < length) & (x[1] > 0) & (x[1] < width))

    def edge_x_1(x):
        return (np.isclose(x[1], 0) & np.isclose(x[2], 0) & (x[0] > 0) & (x[0] < length))
    def edge_x_2(x):
        return (np.isclose(x[1], width) & np.isclose(x[2], 0) & (x[0] > 0) & (x[0] < length))
    def edge_x_3(x):
        return (np.isclose(x[1], width) & np.isclose(x[2], height) & (x[0] > 0) & (x[0] < length))
    def edge_x_4(x):
        return (np.isclose(x[1], 0) & np.isclose(x[2], height) & (x[0] > 0) & (x[0] < length))
    def edge_y_1(x):
        return (np.isclose(x[0], 0) & np.isclose(x[2], 0) & (x[1] > 0) & (x[1] < width))
    def edge_y_2(x):
        return (np.isclose(x[0], length) & np.isclose(x[2], 0) & (x[1] > 0) & (x[1] < width))
    def edge_y_3(x):
        return (np.isclose(x[0], length) & np.isclose(x[2], height) & (x[1] > 0) & (x[1] < width)) 
    def edge_y_4(x):
        return (np.isclose(x[0], 0) & np.isclose(x[2], height) & (x[1] > 0) & (x[1] < width))
    def edge_z_1(x):
        return (np.isclose(x[0], 0) & np.isclose(x[1], 0) & (x[2] > 0) & (x[2] < height))
    def edge_z_2(x):
        return (np.isclose(x[0], length) & np.isclose(x[1], 0) & (x[2] > 0) & (x[2] < height))
    def edge_z_3(x):
        return (np.isclose(x[0], length) & np.isclose(x[1], width) & (x[2] > 0) & (x[2] < height))
    def edge_z_4(x):
        return (np.isclose(x[0], 0) & np.isclose(x[1], width) & (x[2] > 0) & (x[2] < height))


    ## Boundary conditions ##

    fdim_point = 0
    fdim_line  = 1
    fdim_plane = 2

    boundary_vertices_facets = mesh.locate_entities_boundary(domain, fdim_point, vertices)

    left_facets   = mesh.locate_entities_boundary(domain, fdim_plane, left)
    right_facets  = mesh.locate_entities_boundary(domain, fdim_plane, right)
    front_facets  = mesh.locate_entities_boundary(domain, fdim_plane, front)
    back_facets   = mesh.locate_entities_boundary(domain, fdim_plane, back)
    bottom_facets = mesh.locate_entities_boundary(domain, fdim_plane, bottom)
    top_facets    = mesh.locate_entities_boundary(domain, fdim_plane, top)

    face_indices = np.hstack([left_facets, right_facets, front_facets, back_facets, bottom_facets, top_facets]).astype(np.int32)
    face_markers = np.hstack([np.full(len(left_facets), 1),
                            np.full(len(right_facets), 2),
                            np.full(len(front_facets), 3),
                            np.full(len(back_facets), 4), 
                            np.full(len(bottom_facets), 5),
                            np.full(len(top_facets), 6)]).astype(np.int32)

    face_tags = mesh.meshtags(domain, fdim_plane, face_indices, face_markers)

    edge_x_1_facets = mesh.locate_entities_boundary(domain, fdim_line, edge_x_1)
    edge_x_2_facets = mesh.locate_entities_boundary(domain, fdim_line, edge_x_2)
    edge_x_3_facets = mesh.locate_entities_boundary(domain, fdim_line, edge_x_3)
    edge_x_4_facets = mesh.locate_entities_boundary(domain, fdim_line, edge_x_4)
    edge_y_1_facets = mesh.locate_entities_boundary(domain, fdim_line, edge_y_1)
    edge_y_2_facets = mesh.locate_entities_boundary(domain, fdim_line, edge_y_2)
    edge_y_3_facets = mesh.locate_entities_boundary(domain, fdim_line, edge_y_3)
    edge_y_4_facets = mesh.locate_entities_boundary(domain, fdim_line, edge_y_4)
    edge_z_1_facets = mesh.locate_entities_boundary(domain, fdim_line, edge_z_1)
    edge_z_2_facets = mesh.locate_entities_boundary(domain, fdim_line, edge_z_2)
    edge_z_3_facets = mesh.locate_entities_boundary(domain, fdim_line, edge_z_3)   
    edge_z_4_facets = mesh.locate_entities_boundary(domain, fdim_line, edge_z_4)

    edge_x_indices = np.hstack([edge_x_1_facets, edge_x_2_facets, edge_x_3_facets, edge_x_4_facets]).astype(np.int32)
    edge_x_markers = np.hstack([np.full(len(edge_x_1_facets), 1),
                            np.full(len(edge_x_2_facets) + len(edge_x_3_facets) + len(edge_x_4_facets), 2)]).astype(np.int32)
    edge_x_tags    = mesh.meshtags(domain, fdim_line, edge_x_indices, edge_x_markers)

    edge_y_indices = np.hstack([edge_y_1_facets, edge_y_2_facets, edge_y_3_facets, edge_y_4_facets]).astype(np.int32)
    edge_y_markers = np.hstack([np.full(len(edge_y_1_facets), 1),
                            np.full(len(edge_y_2_facets) + len(edge_y_3_facets) + len(edge_y_4_facets), 2)]).astype(np.int32)
    edge_y_tags    = mesh.meshtags(domain, fdim_line, edge_y_indices, edge_y_markers)

    edge_z_indices = np.hstack([edge_z_1_facets, edge_z_2_facets, edge_z_3_facets, edge_z_4_facets]).astype(np.int32)
    edge_z_markers = np.hstack([np.full(len(edge_z_1_facets), 1),
                            np.full(len(edge_z_2_facets) + len(edge_z_3_facets) + len(edge_z_4_facets), 2)]).astype(np.int32)
    edge_z_tags    = mesh.meshtags(domain, fdim_line, edge_z_indices, edge_z_markers)


    ## Define displacement function spaces ##

    S_disp = fem.functionspace(domain, ("Lagrange", 1, (domain.geometry.dim, )))


    ## Apply the periodic boundary conditions for displacement ##

    def periodic_x(x):
        out_x = np.copy(x)
        out_x[0] = x[0] + (length)
        return out_x
    def periodic_y(x):
        out_x = np.copy(x)
        out_x[1] = x[1] + (width)
        return out_x
    def periodic_z(x):
        out_x = np.copy(x)
        out_x[2] = x[2] + (height)
        return out_x

    def periodic_edge_x(x):
        out_x = np.copy(x)
        out_x[1] = x[1] - (width) * edge_x_2(x) - (width) * edge_x_3(x)
        out_x[2] = x[2] - (height) * edge_x_4(x) - (height) * edge_x_3(x)
        return out_x
    def periodic_edge_y(x):
        out_x = np.copy(x)
        out_x[0] = x[0] - (length) * edge_y_2(x) - (length) * edge_y_3(x)
        out_x[2] = x[2] - (height) * edge_y_4(x) - (height) * edge_y_3(x)
        return out_x
    def periodic_edge_z(x):
        out_x = np.copy(x)
        out_x[0] = x[0] - (length) * edge_z_2(x) - (length) * edge_z_3(x)
        out_x[1] = x[1] - (width) * edge_z_4(x) - (width) * edge_z_3(x)
        return out_x

    bcs_disp = fem.dirichletbc(np.array([0, 0, 0], dtype=default_scalar_type), fem.locate_dofs_topological(S_disp, fdim_point, boundary_vertices_facets), S_disp)

    mpc = dolfinx_mpc.MultiPointConstraint(S_disp)
    mpc.create_periodic_constraint_topological(
        S_disp, face_tags, 1, periodic_x, [bcs_disp]
    )
    mpc.create_periodic_constraint_topological(
        S_disp, face_tags, 3, periodic_y, [bcs_disp]
    )
    mpc.create_periodic_constraint_topological(
        S_disp, face_tags, 5, periodic_z, [bcs_disp]
    )
    mpc.create_periodic_constraint_topological(
        S_disp, edge_x_tags, 2, periodic_edge_x, [bcs_disp]
    )
    mpc.create_periodic_constraint_topological(
        S_disp, edge_y_tags, 2, periodic_edge_y, [bcs_disp]
    )
    mpc.create_periodic_constraint_topological(
        S_disp, edge_z_tags, 2, periodic_edge_z, [bcs_disp]
    )
    mpc.finalize()

    h = ufl.TrialFunction(S_disp)
    k = ufl.TestFunction(S_disp)


    ## Define the (linear) variational problem for displacement ##

    def get_voigt(matrix):
        return ufl.as_vector([matrix[0, 0],   # ₁₁
                            matrix[1, 1],   # ₂₂
                            matrix[2, 2],   # ₃₃
                            2 * matrix[1, 2],   # ₂₃ 
                            2 * matrix[0, 2],   # ₁₃ 
                            2 * matrix[0, 1]    # ₁₂ 
                            ])

    def epsilon_sym(u):
        epsilon = ufl.sym(ufl.grad(u))
        return get_voigt(epsilon)

    def get_stiffness_matrix(mu, lam):
        C = ufl.as_matrix([[lam + 2 * mu,     lam,           lam,           0,              0,              0],
                        [lam,           lam + 2 * mu,     lam,           0,              0,              0],
                        [lam,               lam,       lam + 2 * mu,     0,              0,              0],
                        [0,                 0,              0,          mu,              0,              0],
                        [0,                 0,              0,           0,             mu,              0],
                        [0,                 0,              0,           0,              0,             mu]])
        return C

    dim_disp = domain.geometry.dim
    I = ufl.variable(ufl.Identity(dim_disp))

    applied_eps = fem.Constant(domain, np.zeros((6)))
    applied_eps_ = fem.Constant(domain, np.zeros((6)))

    # def epsilon_volume(r):
    #     beta = -0.3 * r ** 2 + 0.22 * r - 0.17
    #     return get_voigt(beta * r * I)

    # def epsilon_thermal(alpha, delta_temp):
    #     return get_voigt(alpha * delta_temp * I)

    def P_tot(h, stiffness_matrix):
        epsilon_tot = applied_eps + epsilon_sym(h)
        sigma_voigt = ufl.dot(stiffness_matrix, epsilon_tot)
        return sigma_voigt

    # delta_temp = u_temp_prev - initial_temp

    a_disp = 0.0

    dx = ufl.Measure("dx", domain=domain, subdomain_data=cell_tags)

    for tag, (mu, lam, alpha) in material_properties.items():
        stiffness_matrix = get_stiffness_matrix(mu, lam)
        # if tag == 2:
        a_tag, L_disp = ufl.system(ufl.inner(P_tot(h, stiffness_matrix), epsilon_sym(k)) * dx(tag))
        a_disp += a_tag
            # L_disp += ufl.inner(epsilon_sym(v_disp_current), P_thermal(mu, lam, alpha, delta_temp)) * dx(tag)
            # L_disp += ufl.inner(epsilon_sym(v_disp_current), P_volume(mu, lam, r_func)) * dx(tag)
        # else:
            # a_disp += ufl.inner(epsilon_sym(v_disp_current), P_tot(u_disp_current, mu, lam)) * dx(tag)
            # L_disp += ufl.inner(epsilon_sym(v_disp_current), P_thermal(mu, lam, alpha, delta_temp)) * dx(tag)

    petsc_options = {
        "ksp_type": "cg",
        "pc_type": "hypre",
        "pc_hypre_type": "boomeramg",
        "ksp_rtol": 1e-6,
        "ksp_max_it": 1000,
        "ksp_reuse_preconditioner": "true"
    }


    ## Initialize the xdmf output file and initial conditions ##

    if not os.path.exists("results"):
        os.makedirs("results")

    xdmf = io.XDMFFile(domain.comm, "results/rve_h_functions.xdmf", "w")
    xdmf.write_mesh(domain)


    ## Define the displacement solver ##

    u_mpc = fem.Function(mpc.function_space)
    problem_disp = MPCLinearProblem(a_disp, L_disp, mpc, bcs=[bcs_disp], u=u_mpc, petsc_options=petsc_options)

    elementary_load = np.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0], 
                                [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                                [0.0, 0.0, 1.0, 0.0, 0.0, 0.0], 
                                [0.0, 0.0, 0.0, 1.0, 0.0, 0.0], 
                                [0.0, 0.0, 0.0, 0.0, 1.0, 0.0], 
                                [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])

    dim_load = elementary_load.shape[0]
    D_homogenized_matrix = np.zeros((dim_load, dim_load))

    for i in range(dim_load):
        func_name = f"h_store_load_{i+1}"
        # globals()[func_name] = fem.Function(S_disp_mpc)
        globals()[func_name] = fem.Function(S_disp)
        globals()[func_name].name = f"h_store_load_{i+1}"

    unit_cell_volume = length * width * height

    # topology, cell_types, geometry = plot.vtk_mesh(S_disp_mpc)
    topology, cell_types, geometry = plot.vtk_mesh(S_disp)
    h_solve_total = np.zeros((geometry.shape[0], dim_load, 3))

    
    ## Time stepping
    
    update_matrix_material_properties()

    # ramp_param = min(max(t / ramp_duration, 0.0), 1.0)
    # temp_bc = initial_temp + ramp_param * (final_temp - initial_temp)
    # u_temp_prev.interpolate(lambda x: np.full(x.shape[1], temp_bc, dtype=default_scalar_type))

    for i in range(dim_load):
        print(f"Applying elementary load {i + 1}...")
        applied_eps.value = elementary_load[i]
        h_solve = problem_disp.solve()
        
        for j in range(dim_load):
            applied_eps_.value = elementary_load[j]
            for tag, (mu, lam, alpha) in material_properties.items():
                stiffness_matrix = get_stiffness_matrix(mu, lam)
                D_homogenized_matrix[i, j] += (1 / unit_cell_volume) * fem.assemble_scalar(fem.form(ufl.inner(P_tot(h_solve, stiffness_matrix), applied_eps_) * dx(tag)))
        
        # store the h solution for this load case
        func_name = f"h_store_load_{i+1}"
        globals()[func_name].x.array[:] = h_solve.x.array[:]
        globals()[func_name].x.scatter_forward()
        
        xdmf.write_function(globals()[func_name])

    print(' ')
    print("Homogenized D matrix:")
    print(' ')
    for i in range(dim_load):
        for j in range(dim_load):
            if D_homogenized_matrix[i, j] < 1e-3:
                print('  0.0  ', end='  ')
            else:
                print(f"{D_homogenized_matrix[i, j]:.3e}", end='  ')

        print('\n', end='  ')
    print(' ')

    xdmf.close()
    print('\n')

    # convert D_homogenized_matrix to ufl form:
    D_homogenized_matrix = ufl.as_matrix(D_homogenized_matrix)

    return D_homogenized_matrix