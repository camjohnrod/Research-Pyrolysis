import numpy as np
import dolfinx_mpc.utils
from   dolfinx import fem, mesh, default_scalar_type


def get_mpc(domain, length, width, height):

    S_disp = fem.functionspace(domain, ("Lagrange", 1, (domain.geometry.dim, )))

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

    def periodic_x(x):
        out_x    = np.copy(x)
        out_x[0] = x[0] + (length)
        return out_x
    def periodic_y(x):
        out_x    = np.copy(x)
        out_x[1] = x[1] + (width)
        return out_x
    def periodic_z(x):
        out_x    = np.copy(x)
        out_x[2] = x[2] + (height)
        return out_x

    def periodic_edge_x(x):
        out_x    = np.copy(x)
        out_x[1] = x[1] - (width) * edge_x_2(x) - (width) * edge_x_3(x)
        out_x[2] = x[2] - (height) * edge_x_4(x) - (height) * edge_x_3(x)
        return out_x
    def periodic_edge_y(x):
        out_x    = np.copy(x)
        out_x[0] = x[0] - (length) * edge_y_2(x) - (length) * edge_y_3(x)
        out_x[2] = x[2] - (height) * edge_y_4(x) - (height) * edge_y_3(x)
        return out_x
    def periodic_edge_z(x):
        out_x    = np.copy(x)
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

    return mpc, bcs_disp