import numpy as np
import time
import ufl
import dolfinx_mpc
import dolfinx_mpc.utils
from   dolfinx_mpc import LinearProblem as MPCLinearProblem
from   dolfinx import fem, default_scalar_type
from   petsc4py import PETSc
from   mpi4py import MPI
from   tqdm.auto import trange


def solve_unit_cell(scale, domain, cell_tags, material_state, mpc, bcs_disp, u_temp_prev,
                    beta_history, stiffness_tensor_homogenized, eigenstrain_homogenized,
                    eigenstrain_microscale):


    ##==========================================================================##
    ##===  SETUP: geometry, averaged material properties, UFL building blocks ===##
    ##==========================================================================##

    rotated_eigenstrain_microscale = fem.Constant(domain, np.zeros(6, dtype=default_scalar_type))
    rotated_eigenstrain_microscale.value[0], rotated_eigenstrain_microscale.value[1], rotated_eigenstrain_microscale.value[2] = eigenstrain_microscale.value[1], eigenstrain_microscale.value[0], eigenstrain_microscale.value[2]
    rotated_eigenstrain_microscale.value[3], rotated_eigenstrain_microscale.value[4], rotated_eigenstrain_microscale.value[5] = eigenstrain_microscale.value[3], eigenstrain_microscale.value[4], eigenstrain_microscale.value[5]

    x_min, x_max = domain.geometry.x[:, 0].min(), domain.geometry.x[:, 0].max()
    y_min, y_max = domain.geometry.x[:, 1].min(), domain.geometry.x[:, 1].max()
    z_min, z_max = domain.geometry.x[:, 2].min(), domain.geometry.x[:, 2].max()
    length           = x_max - x_min
    width            = y_max - y_min
    height           = z_max - z_min
    unit_cell_volume = length * width * height

    E1_avg      = np.mean(material_state.E1.x.array[:])
    E2_avg      = np.mean(material_state.E2.x.array[:])
    E3_avg      = np.mean(material_state.E3.x.array[:])
    nu12_avg    = np.mean(material_state.nu12.x.array[:])
    nu13_avg    = np.mean(material_state.nu13.x.array[:])
    nu23_avg    = np.mean(material_state.nu23.x.array[:])
    G12_avg     = np.mean(material_state.G12.x.array[:])
    G13_avg     = np.mean(material_state.G13.x.array[:])
    G23_avg     = np.mean(material_state.G23.x.array[:])
    alpha_avg   = fem.Constant(domain, np.mean(material_state.alpha.x.array[:]))
    vf_poly_avg = fem.Constant(domain, np.mean(material_state.vf_poly))
    r_new_avg   = fem.Constant(domain, np.mean(material_state.r_new.x.array[:]))
    r_old_avg   = fem.Constant(domain, np.mean(material_state.r_old.x.array[:]))

    S  = fem.functionspace(domain, ("Lagrange", 1, (domain.geometry.dim, )))
    h  = ufl.TrialFunction(S)
    h_ = ufl.TestFunction(S)
    k  = ufl.TrialFunction(S)
    k_ = ufl.TestFunction(S)

    def get_voigt(tensor):
        return ufl.as_vector([   tensor[0, 0],
                                 tensor[1, 1],
                                 tensor[2, 2],
                             2 * tensor[1, 2],
                             2 * tensor[0, 2],
                             2 * tensor[0, 1]])

    def epsilon_sym(u):
        return get_voigt(ufl.sym(ufl.grad(u)))

    dim = domain.geometry.dim
    I   = ufl.variable(ufl.Identity(dim))

    def get_beta(r_new, r_old):
        return vf_poly_avg * ((-0.3 * r_new**3 + 0.22 * r_new**2 - 0.17 * r_new)
                             - (-0.3 * r_old**3 + 0.22 * r_old**2 - 0.17 * r_old))

    def epsilon_volume(beta_current, beta_history):
        return get_voigt((beta_current + beta_history) * I)

    def epsilon_thermal(alpha, delta_temp):
        return get_voigt(alpha * delta_temp * I)

    def get_stiffness_tensor(E1, E2, E3, nu12, nu13, nu23, G12, G13, G23):
        S_mat = np.array([[1/E1,     -nu12/E1, -nu13/E1, 0,      0,      0     ],
                          [-nu12/E1,  1/E2,    -nu23/E2, 0,      0,      0     ],
                          [-nu13/E1, -nu23/E2,  1/E3,    0,      0,      0     ],
                          [0,         0,         0,      1/G23,  0,      0     ],
                          [0,         0,         0,       0,     1/G13,  0     ],
                          [0,         0,         0,       0,      0,     1/G12 ]])
        return fem.Constant(domain, np.linalg.inv(S_mat))

    applied_eps  = fem.Constant(domain, np.zeros(6))
    applied_eps_ = fem.Constant(domain, np.zeros(6))

    def P_tot_multiple_rhs(h, C):
        return ufl.dot(C, applied_eps + epsilon_sym(h))

    def P_tot(k, C):
        return ufl.dot(C, epsilon_sym(k))


    ##==========================================================================##
    ##===  BUILD UFL FORMS                                                    ===##
    ##==========================================================================##

    if scale == 'micro':
        material_properties = {
            1: (E1_avg, E2_avg, E3_avg, nu12_avg, nu13_avg, nu23_avg, G12_avg, G13_avg, G23_avg, alpha_avg),
            2: (material_state.fiber.E1, material_state.fiber.E2, material_state.fiber.E3,
                material_state.fiber.nu12, material_state.fiber.nu13, material_state.fiber.nu23,
                material_state.fiber.G12, material_state.fiber.G13, material_state.fiber.G23,
                material_state.fiber.alpha)
        }
    elif scale == 'meso':
        nu21 = material_state.fiber.nu12 * material_state.fiber.E2 / material_state.fiber.E1
        material_properties = {
            1: (E1_avg, E2_avg, E3_avg, nu12_avg, nu13_avg, nu23_avg, G12_avg, G13_avg, G23_avg, alpha_avg),
            2: (material_state.fiber.E1, material_state.fiber.E2, material_state.fiber.E3,
                material_state.fiber.nu12, material_state.fiber.nu13, material_state.fiber.nu23,
                material_state.fiber.G12, material_state.fiber.G13, material_state.fiber.G23,
                material_state.fiber.alpha),
            3: (material_state.fiber.E2, material_state.fiber.E1, material_state.fiber.E3,
                nu21, material_state.fiber.nu23, material_state.fiber.nu13,
                material_state.fiber.G12, material_state.fiber.G13, material_state.fiber.G23,
                material_state.fiber.alpha)
        }

    stiffness_matrices = {}
    for tag, props in material_properties.items():
        E1_p, E2_p, E3_p, nu12_p, nu13_p, nu23_p, G12_p, G13_p, G23_p, _ = props
        stiffness_matrices[tag] = get_stiffness_tensor(E1_p, E2_p, E3_p, nu12_p, nu13_p,
                                                        nu23_p, G12_p, G13_p, G23_p)

    dx           = ufl.Measure("dx", domain=domain, subdomain_data=cell_tags)
    a_h          = 0.0
    L_h          = 0.0
    a_k          = 0.0
    L_k          = 0.0
    delta_temp   = fem.Constant(domain, u_temp_prev.x.array[:].mean() - 200.0)
    beta_current = get_beta(r_new_avg, r_old_avg)

    for tag, (E1, E2, E3, nu12, nu13, nu23, G12, G13, G23, alpha) in material_properties.items():
        C = stiffness_matrices[tag]
        a_h_tag, L_h_tag = ufl.system(ufl.inner(P_tot_multiple_rhs(h, C), epsilon_sym(h_)) * dx(tag))
        a_h += a_h_tag
        L_h += L_h_tag
        if tag > 1:
            a_k += ufl.inner(epsilon_sym(k_), P_tot(k, C)) * dx(tag)
            if tag == 2:
                L_k += ufl.inner(epsilon_sym(k_), ufl.dot(C, epsilon_thermal(alpha, delta_temp) + eigenstrain_microscale)) * dx(tag)
            elif tag == 3:
                L_k += ufl.inner(epsilon_sym(k_), ufl.dot(C, epsilon_thermal(alpha, delta_temp) + rotated_eigenstrain_microscale)) * dx(tag)
        else:
            a_k += ufl.inner(epsilon_sym(k_), P_tot(k, C)) * dx(tag)
            L_k += ufl.inner(epsilon_sym(k_), ufl.dot(C, epsilon_thermal(alpha, delta_temp) + epsilon_volume(beta_current, beta_history))) * dx(tag)

    
    # ping

    # u_mpc_h = fem.Function(mpc.function_space)
    # u_mpc_k = fem.Function(mpc.function_space)

    # # petsc_options={}
    # # petsc_options = {
    # # "ksp_type": "preonly",
    # # "pc_type": "lu", # or "cholesky" if SPD
    # # "pc_factor_mat_solver_type": "umfpack" # or "umfpack"/"cholmod"
    # # }
    # petsc_options = {
    #     "ksp_type": "gmres",
    #     "pc_type": "ilu"
    # }

    # problem_h = MPCLinearProblem(a_h, L_h, mpc, bcs=[bcs_disp], u=u_mpc_h, petsc_options=petsc_options)
    # problem_k = MPCLinearProblem(a_k, L_k, mpc, bcs=[bcs_disp], u=u_mpc_k, petsc_options=petsc_options)
    # K_solve   = problem_k.solve()
    
    # elementary_load = np.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0], 
    #                             [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
    #                             [0.0, 0.0, 1.0, 0.0, 0.0, 0.0], 
    #                             [0.0, 0.0, 0.0, 1.0, 0.0, 0.0], 
    #                             [0.0, 0.0, 0.0, 0.0, 1.0, 0.0], 
    #                             [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])

    # dim_load = elementary_load.shape[0]
    # temporary_tensor = np.zeros((dim_load, dim_load))
    # vol_inv = 1.0 / unit_cell_volume
    # j_allowed = {i: [j for j in range(dim_load) if not (((i > 2) or (j > 2)) and (i != j))] for i in range(dim_load)}

    # for i in trange(dim_load, colour="red", desc=f"Solving Unit Cell", position=1, leave=False, bar_format='{l_bar}{bar:30}{r_bar}', total=dim_load):
    # # for i in range(dim_load):
    #     applied_eps.value = elementary_load[i]
    #     H_solve = problem_h.solve()

    #     for j in j_allowed[i]:
    #         applied_eps_.value = elementary_load[j]
    #         for tag, stiffness_matrix in stiffness_matrices.items():
    #             temporary_tensor[i, j] += vol_inv * fem.assemble_scalar(fem.form(ufl.inner(P_tot_multiple_rhs(H_solve, stiffness_matrix), applied_eps_) * dx(tag)))

    # stiffness_tensor_homogenized.value = temporary_tensor

    # temporary_vector = np.zeros((dim_load))
    # for tag, (E1, E2, E3, nu12, nu13, nu23, G12, G13, G23, alpha) in material_properties.items():
    #     stiffness_matrix = stiffness_matrices[tag]
    #     for j in range(dim_load):
    #         if j > 2:
    #             continue
    #         applied_eps_.value = elementary_load[j]
    #         if tag > 1:
    #             temporary_vector[j] += fem.assemble_scalar(fem.form(ufl.inner(ufl.dot(stiffness_matrix, epsilon_sym(K_solve)), applied_eps_) * dx(tag)))
    #             if tag == 2:
    #                 temporary_vector[j] -= fem.assemble_scalar(fem.form(ufl.inner(ufl.dot(stiffness_matrix, epsilon_thermal(alpha, delta_temp) + eigenstrain_microscale), applied_eps_) * dx(tag)))
    #             elif tag == 3:
    #                 temporary_vector[j] -= fem.assemble_scalar(fem.form(ufl.inner(ufl.dot(stiffness_matrix, epsilon_thermal(alpha, delta_temp) + rotated_eigenstrain_microscale), applied_eps_) * dx(tag)))
    #         else:
    #             temporary_vector[j] += fem.assemble_scalar(fem.form(ufl.inner(ufl.dot(stiffness_matrix, epsilon_sym(K_solve)), applied_eps_) * dx(tag)))
    #             temporary_vector[j] -= fem.assemble_scalar(fem.form(ufl.inner(ufl.dot(stiffness_matrix, epsilon_thermal(alpha, delta_temp) + epsilon_volume(beta_current, beta_history)), applied_eps_) * dx(tag)))
                
    # eigenstrain_bar = -(1 / unit_cell_volume) * np.linalg.inv(temporary_tensor) @ temporary_vector
    # eigenstrain_homogenized.value = eigenstrain_bar

    # beta_history.value += float(beta_current)

    # ping


    # peng 

    ##==========================================================================##
    ##===  LU OPTIMIZATION: assemble shared stiffness matrix and factor once ===##
    ##===                                                                     ===##
    ##===  a_h and a_k share the same bilinear part (standard elasticity).   ===##
    ##===  All 7 solves (6 stiffness columns + 1 eigenstrain) reuse one       ===##
    ##===  LU factorization instead of factoring separately per solve.        ===##
    ##==========================================================================##

    a_form   = fem.form(a_h)
    L_h_form = fem.form(L_h)
    L_k_form = fem.form(L_k)

    t0 = time.perf_counter()
    
    A = dolfinx_mpc.assemble_matrix(a_form, mpc, bcs=[bcs_disp])
    A.assemble()

    ksp = PETSc.KSP().create(domain.comm)
    ksp.setOperators(A)
    ksp.setType("preonly")       # direct solve — no Krylov iterations
    ksp.getPC().setType("lu")    # LU factorization
    ksp.getPC().setFactorSolverType("umfpack")
    ksp.setFromOptions()
    ksp.setUp()                  # ← factorization happens here (expensive, done once)

    # print(f"  [LU ] ({scale}) Factorization:   {time.perf_counter() - t0:.4f}s")

    # ── Helpers ───────────────────────────────────────────────────────────────

    def assemble_rhs(L_form):
        """Assemble a RHS vector with MPC constraints and Dirichlet BCs applied."""
        b = dolfinx_mpc.assemble_vector(L_form, mpc)
        dolfinx_mpc.apply_lifting(b, [a_form], [[bcs_disp]], mpc)
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        fem.petsc.set_bc(b, [bcs_disp])
        return b

    def mpc_solve(b):
        """Back-substitute only — the factorization in ksp is already done."""
        x = A.createVecRight()
        ksp.solve(b, x)
        mpc.backsubstitution(x)
        u = fem.Function(mpc.function_space)
        u.x.array[:] = x.array_r
        u.x.scatter_forward()
        return u

    # ── Eigenstrain solve — uses the same factored ksp ────────────────────────

    t0      = time.perf_counter()
    K_solve = mpc_solve(assemble_rhs(L_k_form))
    # print(f"  [LU ] ({scale}) Eigenstrain solve: {time.perf_counter() - t0:.4f}s")


    ##==========================================================================##
    ##===  BLOCK SOLVER: solve all 6 RHS columns simultaneously              ===##
    ##===                                                                     ===##
    ##===  Instead of calling ksp.solve(b, x) six times sequentially, we     ===##
    ##===  pack all 6 RHS vectors into a dense matrix B and call             ===##
    ##===  ksp.matSolve(B, X). PETSc then performs the 6 back-substitutions  ===##
    ##===  in one BLAS-3 pass, loading L and U from memory only once.        ===##
    ##==========================================================================##

    elementary_load = np.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                                [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                                [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])

    dim_load = elementary_load.shape[0]

    t0 = time.perf_counter()

    # ── Step 1: assemble all 6 RHS vectors ────────────────────────────────────
    rhs_list = []
    for i in range(dim_load):
        applied_eps.value = elementary_load[i]
        rhs_list.append(assemble_rhs(L_h_form))

    # ── Step 2: pack RHS vectors into a PETSc dense matrix B (n_dof × 6) ──────
    # Each column of B is one load case's RHS vector.
    # Rows are distributed across MPI ranks; columns (6) are local to each rank.
    n_local  = rhs_list[0].local_size
    n_global = rhs_list[0].size
    rstart   = A.getOwnershipRange()[0]   # global index of first local row

    B = PETSc.Mat()
    B.createDense(((n_local, n_global), (dim_load, dim_load)), comm=domain.comm)
    B.setUp()

    for i, b_i in enumerate(rhs_list):
        local_rows = list(range(rstart, rstart + n_local))
        B.setValues(local_rows, [i], b_i.array_r[:n_local].reshape(-1, 1))
    B.assemblyBegin()
    B.assemblyEnd()

    # ── Step 3: solve AX = B (all 6 columns at once via BLAS-3) ──────────────
    X = PETSc.Mat()
    X.createDense(((n_local, n_global), (dim_load, dim_load)), comm=domain.comm)
    X.setUp()

    ksp.matSolve(B, X)   # ← single BLAS-3 call replaces 6 sequential ksp.solve calls

    # print(f"  [BLK] ({scale}) Block solve (6 cols): {time.perf_counter() - t0:.4f}s")

    # ── Step 4: extract solution columns, apply MPC backsubstitution ──────────
    # getDenseArray() returns the local portion of X as a numpy array (n_local × 6)
    X_array  = X.getDenseArray()
    H_solves = []

    for i in range(dim_load):
        x_col = A.createVecRight()
        x_col.array[:n_local] = X_array[:, i]
        mpc.backsubstitution(x_col)
        u = fem.Function(mpc.function_space)
        u.x.array[:] = x_col.array_r
        u.x.scatter_forward()
        H_solves.append(u)

    B.destroy()
    X.destroy()


    # ##==========================================================================##
    # ##===  POST-PROCESSING: assemble homogenized stiffness tensor            ===##
    # ##==========================================================================##

    # temporary_tensor = np.zeros((dim_load, dim_load))
    # vol_inv  = 1.0 / unit_cell_volume
    # j_allowed = {i: [j for j in range(dim_load)
    #                  if not (((i > 2) or (j > 2)) and (i != j))]
    #              for i in range(dim_load)}

    # for i in trange(dim_load, colour="red", desc=f"Assembling C_hom ({scale})",
    #                 position=1, leave=False,
    #                 bar_format='{l_bar}{bar:30}{r_bar}', total=dim_load):
    # # for i in range(dim_load):
    #     applied_eps.value = elementary_load[i]   # needed for P_tot_multiple_rhs evaluation
    #     H_solve = H_solves[i]

    #     for j in j_allowed[i]:
    #         applied_eps_.value = elementary_load[j]
    #         for tag, C in stiffness_matrices.items():
    #             temporary_tensor[i, j] += vol_inv * fem.assemble_scalar(
    #                 fem.form(ufl.inner(P_tot_multiple_rhs(H_solve, C), applied_eps_) * dx(tag)))

    # stiffness_tensor_homogenized.value = temporary_tensor

    # ##==========================================================================##
    # ##===  POST-PROCESSING: assemble homogenized eigenstrain                 ===##
    # ##==========================================================================##

    # temporary_vector = np.zeros(dim_load)

    # for tag, (E1, E2, E3, nu12, nu13, nu23, G12, G13, G23, alpha) in material_properties.items():
    #     C = stiffness_matrices[tag]
    #     for j in range(dim_load):
    #         if j > 2:
    #             continue
    #         applied_eps_.value = elementary_load[j]
    #         if tag > 1:
    #             temporary_vector[j] += fem.assemble_scalar(fem.form(
    #                 ufl.inner(ufl.dot(C, epsilon_sym(K_solve)), applied_eps_) * dx(tag)))
    #             if tag == 2:
    #                 temporary_vector[j] -= fem.assemble_scalar(fem.form(
    #                     ufl.inner(ufl.dot(C, epsilon_thermal(alpha, delta_temp) + eigenstrain_microscale), applied_eps_) * dx(tag)))
    #             elif tag == 3:
    #                 temporary_vector[j] -= fem.assemble_scalar(fem.form(
    #                     ufl.inner(ufl.dot(C, epsilon_thermal(alpha, delta_temp) + rotated_eigenstrain_microscale), applied_eps_) * dx(tag)))
    #         else:
    #             temporary_vector[j] += fem.assemble_scalar(fem.form(
    #                 ufl.inner(ufl.dot(C, epsilon_sym(K_solve)), applied_eps_) * dx(tag)))
    #             temporary_vector[j] -= fem.assemble_scalar(fem.form(
    #                 ufl.inner(ufl.dot(C, epsilon_thermal(alpha, delta_temp) + epsilon_volume(beta_current, beta_history)), applied_eps_) * dx(tag)))

    # eigenstrain_bar              = -(1 / unit_cell_volume) * np.linalg.inv(temporary_tensor) @ temporary_vector
    # eigenstrain_homogenized.value = eigenstrain_bar
    # beta_history.value           += float(beta_current)

    # ##==========================================================================##
    # ##===  CLEANUP                                                            ===##
    # ##==========================================================================##

    # ksp.destroy()
    # A.destroy()

    # peng

    ##==========================================================================##
    ##===  FORM PRE-COMPILATION                                               ===##
    ##===                                                                     ===##
    ##===  fem.form() triggers FFCx JIT compilation — generates C code,      ===##
    ##===  compiles it, and links it. Calling it inside a loop pays this      ===##
    ##===  cost repeatedly for what is purely a numerical change (new         ===##
    ##===  coefficient values). Pre-compiling once and updating coefficient   ===##
    ##===  arrays before each assemble_scalar call avoids all recompilation.  ===##
    ##==========================================================================##

    # Placeholder Functions — compiled forms reference these objects.
    # Updating .x.array[:] before assembly uses the new values without recompiling.
    H_placeholder = fem.Function(mpc.function_space)
    K_placeholder = fem.Function(mpc.function_space)

    # ── Forms for temporary_tensor (stiffness columns) ────────────────────────
    # One form per tag. applied_eps and applied_eps_ are Constants — updating
    # their .value before assembly is sufficient, no recompilation needed.
    forms_stiffness = {
        tag: fem.form(ufl.inner(P_tot_multiple_rhs(H_placeholder, C), applied_eps_) * dx(tag))
        for tag, C in stiffness_matrices.items()
    }

    # ── Forms for temporary_vector (eigenstrain) — K_solve dependent ─────────
    # One form per tag for the epsilon_sym(K_solve) terms.
    forms_eigenstrain_k = {
        tag: fem.form(ufl.inner(ufl.dot(C, epsilon_sym(K_placeholder)), applied_eps_) * dx(tag))
        for tag, C in stiffness_matrices.items()
    }

    # ── Forms for temporary_vector (eigenstrain) — static thermal/chemical ───
    # These don't depend on K_solve or H_solve — only on Constants
    # (alpha, delta_temp, eigenstrain_microscale, beta_current, beta_history).
    # Updating those Constants' .value before assembly handles all variation.
    forms_eigenstrain_static = {}
    for tag, (E1, E2, E3, nu12, nu13, nu23, G12, G13, G23, alpha) in material_properties.items():
        C = stiffness_matrices[tag]
        if tag == 2:
            forms_eigenstrain_static[tag] = fem.form(
                ufl.inner(ufl.dot(C, epsilon_thermal(alpha, delta_temp) + eigenstrain_microscale),
                          applied_eps_) * dx(tag))
        elif tag == 3:
            forms_eigenstrain_static[tag] = fem.form(
                ufl.inner(ufl.dot(C, epsilon_thermal(alpha, delta_temp) + rotated_eigenstrain_microscale),
                          applied_eps_) * dx(tag))
        else:   # tag == 1 (matrix)
            forms_eigenstrain_static[tag] = fem.form(
                ufl.inner(ufl.dot(C, epsilon_thermal(alpha, delta_temp) + epsilon_volume(beta_current, beta_history)),
                          applied_eps_) * dx(tag))

    # Copy K_solve into placeholder once — it doesn't change during post-processing
    K_placeholder.x.array[:] = K_solve.x.array[:]
    K_placeholder.x.scatter_forward()


    ##==========================================================================##
    ##===  POST-PROCESSING: assemble homogenized stiffness tensor            ===##
    ##==========================================================================##

    temporary_tensor = np.zeros((dim_load, dim_load))
    vol_inv   = 1.0 / unit_cell_volume
    j_allowed = {i: [j for j in range(dim_load)
                     if not (((i > 2) or (j > 2)) and (i != j))]
                 for i in range(dim_load)}

    # for i in trange(dim_load, colour="red", desc=f"Assembling C_hom ({scale})",
    #                 position=1, leave=False,
    #                 bar_format='{l_bar}{bar:30}{r_bar}', total=dim_load):
    for i in range(dim_load):

        # Update H_placeholder with solution for load case i — no recompilation
        H_placeholder.x.array[:] = H_solves[i].x.array[:]
        H_placeholder.x.scatter_forward()
        applied_eps.value = elementary_load[i]

        for j in j_allowed[i]:
            applied_eps_.value = elementary_load[j]
            for tag in stiffness_matrices:
                temporary_tensor[i, j] += vol_inv * fem.assemble_scalar(forms_stiffness[tag])

    stiffness_tensor_homogenized.value = temporary_tensor


    ##==========================================================================##
    ##===  POST-PROCESSING: assemble homogenized eigenstrain                 ===##
    ##==========================================================================##

    temporary_vector = np.zeros(dim_load)

    for tag in material_properties:
        for j in range(dim_load):
            if j > 2:
                continue
            applied_eps_.value = elementary_load[j]

            # K_placeholder already holds K_solve values — just assemble
            temporary_vector[j] += fem.assemble_scalar(forms_eigenstrain_k[tag])

            # Static thermal/chemical term — Constants already hold correct values
            temporary_vector[j] -= fem.assemble_scalar(forms_eigenstrain_static[tag])

    eigenstrain_bar               = -(1 / unit_cell_volume) * np.linalg.inv(temporary_tensor) @ temporary_vector
    eigenstrain_homogenized.value = eigenstrain_bar
    beta_history.value           += float(beta_current)


    ##==========================================================================##
    ##===  CLEANUP                                                            ===##
    ##==========================================================================##

    ksp.destroy()
    A.destroy()