import numpy as np
from dolfinx import fem
import dolfinx.mesh as dmesh
import math

_Ri    = 30.0e-3
_Ro    = 37.0e-3
_theta = math.radians(54.0)
_half  = _theta / 2.0

_a1    = math.pi / 2.0 - _half          # original arc start angle
_a2    = math.pi / 2.0 + _half          # original arc end angle
_rot_angle = math.pi / 2.0 - _a2        # rotation applied in Part 6 of mesh script

# Arc angle range in the ROTATED frame (after Part 6 rotation)
_arc_start = _a1 + _rot_angle            # = pi/2 - theta  ≈ 36°
_arc_end   = _a2 + _rot_angle            # = pi/2          = 90°

# Fiber angle in the arm region (constant — arm is straight)
_arm_fiber_angle = _arc_start - math.pi / 2.0   # = -theta ≈ -54°

# Arc center position in rotated frame (was at origin, rotated about origin)
_arc_cx = 0.0
_arc_cy = 0.0

# Small tolerance for classifying elements near the boundary
_tol = 0.5e-3   # 0.5 mm — adjust if elements are misclassified


def _fiber_angle_from_xy(x_coord, y_coord):
    """
    Returns the local fiber orientation angle (radians) relative to the
    global +x axis, based on the element centroid position (x, y).

    Classification:
      - Fillet region : Ri <= dist_from_arc_center <= Ro AND
                        arc_start <= atan2(y,x) <= arc_end
        → fiber angle = atan2(y, x) - pi/2   (tangent to the arc, CW direction)

      - Arm region (everything else):
        → fiber angle = -theta  (constant, arm is straight)

    Note: x alone is insufficient because arm and fillet elements share
    overlapping x ranges near the junction.
    """
    dx   = x_coord - _arc_cx
    dy   = y_coord - _arc_cy
    dist = math.sqrt(dx**2 + dy**2)

    in_radial_band = (_Ri - _tol) <= dist <= (_Ro + _tol)

    if in_radial_band:
        phi = math.atan2(dy, dx)
        # Normalise to [0, 2pi) for comparison
        if phi < 0:
            phi += 2.0 * math.pi
        arc_start_norm = _arc_start % (2.0 * math.pi)
        arc_end_norm   = _arc_end   % (2.0 * math.pi)

        in_arc_span = arc_start_norm - _tol / _Ro <= phi <= arc_end_norm + _tol / _Ro

        if in_arc_span:
            # Fillet: fiber tangent to the arc in the CW travel direction
            return math.atan2(dy, dx) - math.pi / 2.0

    # Arm: constant fiber direction
    return _arm_fiber_angle


def bond_matrix_z(theta):
    # Voigt order: [11, 22, 33, 23, 13, 12]

    c = np.cos(theta)
    s = np.sin(theta)

    M = np.array([
        [ c*c,   s*s,  0,   0,   0,  -2*c*s   ],
        [ s*s,   c*c,  0,   0,   0,   2*c*s   ],
        [ 0,     0,    1,   0,   0,   0       ],
        [ 0,     0,    0,   c,   s,   0       ],
        [ 0,     0,    0,  -s,   c,   0       ],
        [ c*s,  -c*s,  0,   0,   0,   c*c-s*s ],
    ])
    return M


def build_spatial_fields(domain, C_hom_value, eig_hom_value):

    dim = domain.topology.dim

    S_stiffness   = fem.functionspace(domain, ("DG", 0, (6, 6)))
    S_eig = fem.functionspace(domain, ("DG", 0, (6,)))

    stiffness_spatial   = fem.Function(S_stiffness)
    eig_spatial = fem.Function(S_eig)

    num_cells   = domain.topology.index_map(dim).size_local
    cell_coords = dmesh.compute_midpoints(domain, dim,
                                          np.arange(num_cells, dtype=np.int32))

    for cell_idx in range(num_cells):
        x_coord = cell_coords[cell_idx, 0]
        y_coord = cell_coords[cell_idx, 1]
        theta   = _fiber_angle_from_xy(x_coord, y_coord)
        M       = bond_matrix_z(theta)

        stiffness_rot   = M @ C_hom_value @ M.T
        eig_rot = M @ eig_hom_value

        stiffness_spatial.x.array[cell_idx * 36 : cell_idx * 36 + 36] = stiffness_rot.flatten()
        eig_spatial.x.array[cell_idx * 6  : cell_idx * 6  + 6 ] = eig_rot

    stiffness_spatial.x.scatter_forward()
    eig_spatial.x.scatter_forward()

    return stiffness_spatial, eig_spatial, S_stiffness, S_eig


def update_spatial_fields(stiffness_spatial, eig_spatial, domain, stiffness_tensor_homogenized, eig_homogenized):

    dim         = domain.topology.dim
    num_cells   = domain.topology.index_map(dim).size_local
    cell_coords = dmesh.compute_midpoints(domain, dim,
                                          np.arange(num_cells, dtype=np.int32))

    for cell_idx in range(num_cells):
        x_coord = cell_coords[cell_idx, 0]
        y_coord = cell_coords[cell_idx, 1]
        theta   = _fiber_angle_from_xy(x_coord, y_coord)
        M       = bond_matrix_z(theta)

        C_rot   = M @ stiffness_tensor_homogenized @ M.T
        eig_rot = M @ eig_homogenized

        stiffness_spatial.x.array[cell_idx * 36 : cell_idx * 36 + 36] = C_rot.flatten()
        eig_spatial.x.array[cell_idx * 6  : cell_idx * 6  + 6 ] = eig_rot

    stiffness_spatial.x.scatter_forward()
    eig_spatial.x.scatter_forward()