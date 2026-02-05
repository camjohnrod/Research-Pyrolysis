import os
import sys
import gmsh
import meshio

# ================================================================
# User Input Parameters
# ================================================================

# --- Outer square dimensions ---
Lx = 24e-6      # width
Ly = 24e-6      # height

# --- Circular inclusion ---
cx = Lx / 2     # circle center x
cy = Ly / 2     # circle center y
r  = 7e-6     # radius

# --- Mesh controls ---
n_outer = 10      # divisions on outer edges
n_inner = 10      # divisions on fiber–matrix connectors
nz      = 3       # number of layers in z extrusion
thick   = Lx      # extrusion thickness

# ================================================================
# Initialize
# ================================================================

gmsh.initialize()
gmsh.model.add("rve_3D")

# ================================================================
# Geometry (same 2D setup)
# ================================================================

p1 = gmsh.model.geo.addPoint(0,   0,   0, 1.0)
p2 = gmsh.model.geo.addPoint(Lx,  0,   0, 1.0)
p3 = gmsh.model.geo.addPoint(0,   Ly,  0, 1.0)
p4 = gmsh.model.geo.addPoint(Lx,  Ly,  0, 1.0)

# Fiber quadrants
p5 = gmsh.model.geo.addPoint(cx - r, cy - r, 0, 1.0)
p6 = gmsh.model.geo.addPoint(cx + r, cy - r, 0, 1.0)
p7 = gmsh.model.geo.addPoint(cx + r, cy + r, 0, 1.0)
p8 = gmsh.model.geo.addPoint(cx - r, cy + r, 0, 1.0)
pc = gmsh.model.geo.addPoint(cx, cy, 0, 1.0)

# Outer lines
l1 = gmsh.model.geo.addLine(p1, p3)
l2 = gmsh.model.geo.addLine(p3, p4)
l3 = gmsh.model.geo.addLine(p4, p2)
l4 = gmsh.model.geo.addLine(p2, p1)

# Circle arcs
a1 = gmsh.model.geo.addCircleArc(p5, pc, p6)
a2 = gmsh.model.geo.addCircleArc(p6, pc, p7)
a3 = gmsh.model.geo.addCircleArc(p7, pc, p8)
a4 = gmsh.model.geo.addCircleArc(p8, pc, p5)

# Connectors
c1 = gmsh.model.geo.addLine(p8, p3)
c2 = gmsh.model.geo.addLine(p5, p1)
c3 = gmsh.model.geo.addLine(p6, p2)
c4 = gmsh.model.geo.addLine(p7, p4)

# ================================================================
# Surfaces
# ================================================================

cl1 = gmsh.model.geo.addCurveLoop([l1, -c1, a4, c2])
s1  = gmsh.model.geo.addPlaneSurface([cl1])

cl2 = gmsh.model.geo.addCurveLoop([l4, -c2, a1, c3])
s2  = gmsh.model.geo.addPlaneSurface([cl2])

cl3 = gmsh.model.geo.addCurveLoop([l3, -c3, a2, c4])
s3  = gmsh.model.geo.addPlaneSurface([cl3])

cl4 = gmsh.model.geo.addCurveLoop([l2, -c4, a3, c1])
s4  = gmsh.model.geo.addPlaneSurface([cl4])

clf = gmsh.model.geo.addCurveLoop([a1, a2, a3, a4])
sf  = gmsh.model.geo.addPlaneSurface([clf])

# ================================================================
# Transfinite Meshing
# ================================================================

for l in [l1, l2, l3, l4, a1, a2, a3, a4]:
    gmsh.model.geo.mesh.setTransfiniteCurve(l, n_outer)

for l in [c1, c2, c3, c4]:
    gmsh.model.geo.mesh.setTransfiniteCurve(l, n_inner)

# Transfinite surface corners
gmsh.model.geo.mesh.setTransfiniteSurface(s1, cornerTags=[p3, p1, p5, p8])
gmsh.model.geo.mesh.setTransfiniteSurface(s2, cornerTags=[p1, p2, p6, p5])
gmsh.model.geo.mesh.setTransfiniteSurface(s3, cornerTags=[p2, p4, p7, p6])
gmsh.model.geo.mesh.setTransfiniteSurface(s4, cornerTags=[p4, p3, p8, p7])
gmsh.model.geo.mesh.setTransfiniteSurface(sf,  cornerTags=[p5, p6, p7, p8])

# Recombine → quads
for s in [s1, s2, s3, s4, sf]:
    gmsh.model.geo.mesh.setRecombine(2, s)

gmsh.model.geo.synchronize()

# ================================================================
# 3D Extrusion
# ================================================================
# Extrude ALL 5 surfaces in one call

out = gmsh.model.geo.extrude(
    [(2, s1), (2, s2), (2, s3), (2, s4), (2, sf)],
    0, 0, thick,
    numElements=[nz],
    recombine=True
)

gmsh.model.geo.synchronize()

# The returned "out" list contains:
#  - new surfaces
#  - new volumes (last entries)

# Extract created 3D volumes:
volumes = [entity[1] for entity in out if entity[0] == 3]

fiber_volume  = volumes[-1]     # last volume is the center inclusion
matrix_volumes = volumes[:-1]   # the rest are the outer 4 volumes

# ================================================================
# Physical Groups (3D)
# ================================================================

gmsh.model.addPhysicalGroup(3, [fiber_volume], 1)
gmsh.model.setPhysicalName(3, 1, "fiber")

gmsh.model.addPhysicalGroup(3, matrix_volumes, 2)
gmsh.model.setPhysicalName(3, 2, "matrix")

# Bottom surface physical group (z = 0)
gmsh.model.addPhysicalGroup(2, [s1, s2, s3, s4, sf], 3)
gmsh.model.setPhysicalName(2, 3, "bottom")

# Top surfaces (extract from extrusion)
top_surfs = [entity[1] for entity in out if entity[0] == 2]
gmsh.model.addPhysicalGroup(2, top_surfs, 4)
gmsh.model.setPhysicalName(2, 4, "top")

# ================================================================
# Mesh Generation
# ================================================================

gmsh.model.mesh.generate(3)
gmsh.write("mesh/rve_3D.msh")

gmsh.fltk.run()

gmsh.finalize()


# ================================================================
# Convert .msh to .xdmf using meshio
# ================================================================

filename_old = "rve_3D.msh"
filename_new = "rve_3D.xdmf"

# Read mesh
msh = meshio.read("mesh/rve_3D.msh")

# Determine available cell blocks and their types in a meshio-version-robust way
cell_blocks = []  # list of (cell_type, data_array)
if hasattr(msh, "cells_dict") and msh.cells_dict:
    # meshio >= 4 provides cells_dict
    for ctype, data in msh.cells_dict.items():
        cell_blocks.append((ctype, data))
else:
    # older meshio: msh.cells is a list of (type, data) tuples or CellBlock objects
    for block in msh.cells:
        if hasattr(block, "type") and hasattr(block, "data"):
            cell_blocks.append((block.type, block.data))
        else:
            # tuple-like (type, data)
            ctype, data = block
            cell_blocks.append((ctype, data))

# Prefer common 3D volumetric cell types
vol_types = ["tetra", "hexahedron", "wedge", "pyramid", "prism"]
selected = None
for vt in vol_types:
    for ctype, data in cell_blocks:
        if ctype == vt:
            selected = (ctype, data)
            break
    if selected:
        break

if selected is None:
    print("Error: no 3D volumetric cell block (tetra/hexahedron/wedge/pyramid/prism) found in mesh.")
    print("Available cell blocks:", [c for c, _ in cell_blocks])
    sys.exit(1)

cell_type, cells = selected
print(f"Found cell type: {cell_type} with {len(cells)} elements")

# Extract gmsh:physical tags for the selected cell block in a meshio-version-robust way
tags = None
if hasattr(msh, "cell_data_dict") and "gmsh:physical" in msh.cell_data_dict:
    phys = msh.cell_data_dict["gmsh:physical"]
    # meshio may store this as a dict mapping cell_type->array, or as a list aligned with msh.cells
    if isinstance(phys, dict):
        tags = phys.get(cell_type)
    elif isinstance(phys, list):
        # need to find the index of the block in msh.cells
        idx = None
        # handle msh.cells being list of CellBlock or (type,data)
        for i, block in enumerate(getattr(msh, "cells", [])):
            btype = block.type if hasattr(block, "type") else block[0]
            if btype == cell_type:
                idx = i
                break
        if idx is not None and idx < len(phys):
            tags = phys[idx]

if tags is not None:
    cell_data = {"gmsh:physical": [tags]}
else:
    cell_data = None

out_mesh = meshio.Mesh(
    points=msh.points,
    cells=[(cell_type, cells)],
    cell_data=cell_data,
    field_data=getattr(msh, "field_data", None),
)

meshio.write(f"mesh/{filename_new}", out_mesh)
print(f"Finished converting {filename_new} from {filename_old}")