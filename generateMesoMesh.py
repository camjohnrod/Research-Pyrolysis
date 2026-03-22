import gmsh
import math

# =========================
# USER PARAMETERS
# =========================
step_file = "weave1.stp"

mesh_size = 0.3          # global mesh size
periodic_x = True
periodic_y = True
periodic_z = True        # usually false for textiles

tol = 1e-6                # geometric tolerance for bbox comparisons

# =========================
# INITIALIZE
# =========================
gmsh.initialize()
gmsh.model.add("weave")

# =========================
# IMPORT GEOMETRY
# =========================
gmsh.model.occ.importShapes(step_file)
gmsh.model.occ.synchronize()

volumes = gmsh.model.getEntities(dim=3)

# =========================
# IDENTIFY DOMAIN VS YARNS
# =========================
def bbox(vol):
    return gmsh.model.getBoundingBox(vol[0], vol[1])

# find largest volume = domain
max_vol = None
max_size = -1

for v in volumes:
    b = bbox(v)
    size = (b[3]-b[0])*(b[4]-b[1])*(b[5]-b[2])
    if size > max_size:
        max_size = size
        max_vol = v

domain = max_vol
yarns = [v for v in volumes if v != domain]

print(f"Domain volume: {domain}")
print(f"Yarn pieces: {len(yarns)}")

# =========================
# FRAGMENT (CRITICAL STEP)
# =========================
# ensures yarns are cut by the domain box
all_objs = [domain] + yarns
gmsh.model.occ.fragment(all_objs, [])
gmsh.model.occ.synchronize()

# re-fetch volumes after fragmentation
volumes = gmsh.model.getEntities(dim=3)

# =========================
# RE-IDENTIFY DOMAIN
# =========================
max_vol = None
max_size = -1

for v in volumes:
    b = bbox(v)
    size = (b[3]-b[0])*(b[4]-b[1])*(b[5]-b[2])
    if size > max_size:
        max_size = size
        max_vol = v

domain = max_vol
yarn_vols = [v for v in volumes if v != domain]

# =========================
# GROUP YARNS (PAIR PIECES)
# =========================
# heuristic: pair by proximity of centroids
def centroid(v):
    b = bbox(v)
    return [(b[0]+b[3])/2, (b[1]+b[4])/2, (b[2]+b[5])/2]

groups = []
unused = yarn_vols.copy()

while unused:
    v = unused.pop(0)
    c1 = centroid(v)

    best = None
    best_dist = 1e9

    for u in unused:
        c2 = centroid(u)
        d = math.dist(c1, c2)
        if d < best_dist:
            best_dist = d
            best = u

    groups.append([v, best])
    unused.remove(best)

# =========================
# PHYSICAL GROUPS
# =========================
gmsh.model.addPhysicalGroup(3, [domain[1]], tag=1)
gmsh.model.setPhysicalName(3, 1, "Matrix")

for i, g in enumerate(groups):
    tags = [v[1] for v in g]
    tag = 10 + i
    gmsh.model.addPhysicalGroup(3, tags, tag=tag)
    gmsh.model.setPhysicalName(3, tag, f"Yarn_{i}")

# =========================
# PERIODIC BOUNDARIES
# =========================
def get_surfaces_in_plane(axis, value):
    surfs = gmsh.model.getEntities(dim=2)
    result = []
    for s in surfs:
        b = gmsh.model.getBoundingBox(s[0], s[1])
        if axis == 0 and abs(b[0]-value) < tol and abs(b[3]-value) < tol:
            result.append(s[1])
        if axis == 1 and abs(b[1]-value) < tol and abs(b[4]-value) < tol:
            result.append(s[1])
        if axis == 2 and abs(b[2]-value) < tol and abs(b[5]-value) < tol:
            result.append(s[1])
    return result

db = bbox(domain)

xmin, xmax = db[0], db[3]
ymin, ymax = db[1], db[4]
zmin, zmax = db[2], db[5]

if periodic_x:
    smin = get_surfaces_in_plane(0, xmin)
    smax = get_surfaces_in_plane(0, xmax)
    gmsh.model.mesh.setPeriodic(2, smax, smin,
        [1,0,0, xmax-xmin,
         0,1,0, 0,
         0,0,1, 0,
         0,0,0, 1])

if periodic_y:
    smin = get_surfaces_in_plane(1, ymin)
    smax = get_surfaces_in_plane(1, ymax)
    gmsh.model.mesh.setPeriodic(2, smax, smin,
        [1,0,0, 0,
         0,1,0, ymax-ymin,
         0,0,1, 0,
         0,0,0, 1])

if periodic_z:
    # make tolerance a bit less strict
    gmsh.option.setNumber("Geometry.Tolerance", 1e-4)

    def boundary_faces_at_z(volume, zref, tol=1e-8):
        faces = []
        for dim, tag in gmsh.model.getBoundary([volume], oriented=False, recursive=False):
            if dim != 2:
                continue
            b = gmsh.model.getBoundingBox(dim, tag)
            if abs(b[2] - zref) < tol and abs(b[5] - zref) < tol:
                faces.append(tag)
        return faces

    # after fragment + synchronize + re-identify domain
    b = gmsh.model.getBoundingBox(domain[0], domain[1])
    zmin, zmax = b[2], b[5]
    thickness = zmax - zmin

    bottom_faces = boundary_faces_at_z(domain, zmin)
    top_faces    = boundary_faces_at_z(domain, zmax)

    print("bottom_faces:", bottom_faces)
    print("top_faces:", top_faces)

    gmsh.model.mesh.setPeriodic(
        2,
        top_faces,
        bottom_faces,
        [
            1, 0, 0, 0,
            0, 1, 0, 0,
            0, 0, 1, thickness,
            0, 0, 0, 1
        ]
    )

# =========================
# MESH
# =========================
gmsh.option.setNumber("Mesh.CharacteristicLengthMin", mesh_size)
gmsh.option.setNumber("Mesh.CharacteristicLengthMax", mesh_size)

gmsh.model.mesh.generate(3)

# =========================
# OUTPUT
# =========================
gmsh.write("weave.msh")

gmsh.fltk.run()
gmsh.finalize()