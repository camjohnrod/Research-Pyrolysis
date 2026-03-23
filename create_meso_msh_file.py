import gmsh
import math

# =========================
# USER PARAMETERS
# =========================
step_file = "higher_vf_weave_61.stp"

mesh_size = 0.2
periodic_x = True
periodic_y = True
periodic_z = True

tol = 1e-6
group_tol = 1e-4     # tolerance used to group split yarn chunks
face_tol = 1e-4      # tolerance for identifying boundary faces

# =========================
# INITIALIZE
# =========================
gmsh.initialize()
gmsh.model.add("weave")

# Set tolerance early
gmsh.option.setNumber("Geometry.Tolerance", group_tol)

# =========================
# IMPORT GEOMETRY
# =========================
gmsh.model.occ.importShapes(step_file)
gmsh.model.occ.synchronize()

volumes = gmsh.model.getEntities(dim=3)

# =========================
# HELPER FUNCTIONS
# =========================
def bbox(vol):
    return gmsh.model.getBoundingBox(vol[0], vol[1])

def volume_size(b):
    return (b[3] - b[0]) * (b[4] - b[1]) * (b[5] - b[2])

def extent(b):
    return (b[3] - b[0], b[4] - b[1], b[5] - b[2])

def dominant_axis(b):
    ex, ey, ez = extent(b)
    return max(range(3), key=lambda i: [ex, ey, ez][i])

def q(x):
    return int(round(x / group_tol))

def union_bbox(dimtags):
    xmin = 1e100
    ymin = 1e100
    zmin = 1e100
    xmax = -1e100
    ymax = -1e100
    zmax = -1e100

    for dt in dimtags:
        b = gmsh.model.getBoundingBox(dt[0], dt[1])
        xmin = min(xmin, b[0])
        ymin = min(ymin, b[1])
        zmin = min(zmin, b[2])
        xmax = max(xmax, b[3])
        ymax = max(ymax, b[4])
        zmax = max(zmax, b[5])

    return xmin, ymin, zmin, xmax, ymax, zmax

def get_boundary_surfaces_on_plane(volumes, axis, value, tol):
    faces = []
    for dim, tag in gmsh.model.getBoundary(volumes, oriented=False, recursive=False):
        if dim != 2:
            continue

        b = gmsh.model.getBoundingBox(dim, tag)

        if axis == 0:
            center = 0.5 * (b[0] + b[3])
        elif axis == 1:
            center = 0.5 * (b[1] + b[4])
        else:
            center = 0.5 * (b[2] + b[5])

        if abs(center - value) < tol:
            faces.append(tag)

    uniq = []
    for f in faces:
        if f not in uniq:
            uniq.append(f)
    return uniq

def periodic_matrix(dx, dy, dz):
    return [
        1, 0, 0, dx,
        0, 1, 0, dy,
        0, 0, 1, dz,
        0, 0, 0, 1
    ]

def inside_bbox(p, b, eps=1e-9):
    return (
        b[0] - eps <= p[0] <= b[3] + eps and
        b[1] - eps <= p[1] <= b[4] + eps and
        b[2] - eps <= p[2] <= b[5] + eps
    )

# =========================
# IDENTIFY DOMAIN VS YARNS
# =========================
max_vol = None
max_size = -1

for v in volumes:
    b = bbox(v)
    size = volume_size(b)
    if size > max_size:
        max_size = size
        max_vol = v

domain = max_vol
yarns = [v for v in volumes if v != domain]

print(f"Domain volume: {domain}")
print(f"Yarn pieces: {len(yarns)}")

# =========================
# GROUP ORIGINAL YARN CHUNKS INTO OVERALL YARNS
# =========================
def yarn_signature(v):
    """
    Group the split chunks of a single yarn together.
    Ignore position along the dominant axis and keep only the transverse bbox.
    """
    b = bbox(v)
    ax = dominant_axis(b)

    if ax == 0:  # x-directed yarn
        return (ax, q(b[1]), q(b[4]), q(b[2]), q(b[5]))
    elif ax == 1:  # y-directed yarn
        return (ax, q(b[0]), q(b[3]), q(b[2]), q(b[5]))
    else:  # z-directed yarn
        return (ax, q(b[0]), q(b[3]), q(b[1]), q(b[4]))

yarn_group_dict = {}
for v in yarns:
    key = yarn_signature(v)
    yarn_group_dict.setdefault(key, []).append(v)

overall_yarn_groups = list(yarn_group_dict.values())
overall_yarn_groups.sort(key=lambda g: (dominant_axis(bbox(g[0])), bbox(g[0])[0], bbox(g[0])[1], bbox(g[0])[2]))

print(f"Detected overall yarn groups: {len(overall_yarn_groups)}")

# Categorize each overall yarn into X or Y based on its dominant axis
yarn_x_groups = []
yarn_y_groups = []

for g in overall_yarn_groups:
    ax = dominant_axis(bbox(g[0]))
    if ax == 0:
        yarn_x_groups.append(g)
    elif ax == 1:
        yarn_y_groups.append(g)
    else:
        raise RuntimeError("Found a yarn whose dominant axis is z. This script expects only x- and y-oriented yarns.")

print(f"Yarn_X groups: {len(yarn_x_groups)}")
print(f"Yarn_Y groups: {len(yarn_y_groups)}")

# =========================
# FRAGMENT
# =========================
# Fragment the domain by all yarn chunks.
# out_map[0] = pieces from the original domain
# out_map[i+1] = pieces from the i-th original yarn chunk
out, out_map = gmsh.model.occ.fragment([domain], yarns)
gmsh.model.occ.synchronize()

# Pieces from the original domain
domain_pieces = [dt for dt in out_map[0] if dt[0] == 3]
print(f"Domain split into {len(domain_pieces)} pieces")

# =========================
# FIX: CLASSIFY FINAL VOLUMES INTO PHYSICAL GROUPS
# =========================
# Use the overall yarn groups as spatial regions. Final fragmented volumes are
# assigned by center-of-mass location, giving disjoint physical groups.
x_boxes = [union_bbox(g) for g in yarn_x_groups]
y_boxes = [union_bbox(g) for g in yarn_y_groups]

matrix_tags = []
x_piece_tags = []
y_piece_tags = []

for dim, tag in gmsh.model.getEntities(3):
    cx, cy, cz = gmsh.model.occ.getCenterOfMass(dim, tag)

    in_x = any(inside_bbox((cx, cy, cz), b) for b in x_boxes)
    in_y = any(inside_bbox((cx, cy, cz), b) for b in y_boxes)

    if in_x and not in_y:
        x_piece_tags.append(tag)
    elif in_y and not in_x:
        y_piece_tags.append(tag)
    elif in_x and in_y:
        # Tie-breaker for overlap regions
        b = gmsh.model.getBoundingBox(dim, tag)
        dx = b[3] - b[0]
        dy = b[4] - b[1]
        if dx >= dy:
            x_piece_tags.append(tag)
        else:
            y_piece_tags.append(tag)
    else:
        matrix_tags.append(tag)

# Enforce strict disjointness
x_piece_tags = sorted(set(x_piece_tags))
y_piece_tags = sorted(set(y_piece_tags) - set(x_piece_tags))
matrix_tags = sorted(set(matrix_tags) - set(x_piece_tags) - set(y_piece_tags))

print(f"Matrix pieces: {len(matrix_tags)}")
print(f"Yarn_X pieces: {len(x_piece_tags)}")
print(f"Yarn_Y pieces: {len(y_piece_tags)}")

print(f"Combined Yarn_X volume pieces: {x_piece_tags}")
print(f"Combined Yarn_Y volume pieces: {y_piece_tags}")

# =========================
# PHYSICAL GROUPS
# =========================
gmsh.model.addPhysicalGroup(3, matrix_tags, tag=1)
gmsh.model.setPhysicalName(3, 1, "Matrix")

if len(x_piece_tags) > 0:
    gmsh.model.addPhysicalGroup(3, x_piece_tags, tag=2)
    gmsh.model.setPhysicalName(3, 2, "Yarn_X")

if len(y_piece_tags) > 0:
    gmsh.model.addPhysicalGroup(3, y_piece_tags, tag=3)
    gmsh.model.setPhysicalName(3, 3, "Yarn_Y")

# =========================
# PERIODIC BOUNDARIES
# =========================
xmin, ymin, zmin, xmax, ymax, zmax = union_bbox(domain_pieces)

Lx = xmax - xmin
Ly = ymax - ymin
Lz = zmax - zmin

print(f"Cell bounds:")
print(f"  x: {xmin} -> {xmax}")
print(f"  y: {ymin} -> {ymax}")
print(f"  z: {zmin} -> {zmax}")

if periodic_x:
    smin_x = get_boundary_surfaces_on_plane(domain_pieces, 0, xmin, face_tol)
    smax_x = get_boundary_surfaces_on_plane(domain_pieces, 0, xmax, face_tol)
    print("X periodic:", smin_x, smax_x)
    gmsh.model.mesh.setPeriodic(2, smax_x, smin_x, periodic_matrix(Lx, 0, 0))

if periodic_y:
    smin_y = get_boundary_surfaces_on_plane(domain_pieces, 1, ymin, face_tol)
    smax_y = get_boundary_surfaces_on_plane(domain_pieces, 1, ymax, face_tol)
    print("Y periodic:", smin_y, smax_y)
    gmsh.model.mesh.setPeriodic(2, smax_y, smin_y, periodic_matrix(0, Ly, 0))

if periodic_z:
    smin_z = get_boundary_surfaces_on_plane(domain_pieces, 2, zmin, face_tol)
    smax_z = get_boundary_surfaces_on_plane(domain_pieces, 2, zmax, face_tol)
    print("Z periodic:", smin_z, smax_z)
    gmsh.model.mesh.setPeriodic(2, smax_z, smin_z, periodic_matrix(0, 0, Lz))

# =========================
# MESH
# =========================
gmsh.option.setNumber("Mesh.CharacteristicLengthMin", mesh_size)
gmsh.option.setNumber("Mesh.CharacteristicLengthMax", mesh_size)
gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)

gmsh.model.mesh.generate(3)

# =========================
# OUTPUT
# =========================
gmsh.write("meso_rve_3D.msh")

gmsh.fltk.run()
gmsh.finalize()