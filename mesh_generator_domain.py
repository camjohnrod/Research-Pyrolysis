import math
import gmsh
import meshio
import sys

gmsh.initialize()
gmsh.model.add("L_beam_one_arm_left")

# ---------------------------
# Parameters
# ---------------------------
Ri = 30.0e-3
Ro = 37.0e-3
theta = math.radians(90 - 54.0)
L_tot = 60.0e-3       # length of the rectangular arm
W = 110.0e-3      # thickness in z

L_bar = L_tot - 27e-3

# Transfinite mesh divisions
n_radial = 5      # divisions along radial direction (inner to outer)
n_arc = 9        # divisions along arc
n_length = 7      # divisions along arm length
n_thickness = 15   # divisions through thickness (z-direction)

half = theta / 2
a1 = math.pi/2 - half  # left end of arc (where arm will extend)
a2 = math.pi/2 + half  # right end of arc


def tangent(a):
    return math.sin(a), -math.cos(a), 0.0

# ---------------------------
# PART 1: Create 2D fillet cross-section at z=0
# ---------------------------
pc = gmsh.model.occ.addPoint(0, 0, 0)
po1 = gmsh.model.occ.addPoint(Ro*math.cos(a1), Ro*math.sin(a1), 0)
po2 = gmsh.model.occ.addPoint(Ro*math.cos(a2), Ro*math.sin(a2), 0)
pi1 = gmsh.model.occ.addPoint(Ri*math.cos(a1), Ri*math.sin(a1), 0)
pi2 = gmsh.model.occ.addPoint(Ri*math.cos(a2), Ri*math.sin(a2), 0)

outer_arc = gmsh.model.occ.addCircleArc(po1, pc, po2)
inner_arc = gmsh.model.occ.addCircleArc(pi2, pc, pi1)
r1 = gmsh.model.occ.addLine(po2, pi2)
r2 = gmsh.model.occ.addLine(pi1, po1)

loop = gmsh.model.occ.addCurveLoop([outer_arc, r1, inner_arc, r2])
fillet_surf = gmsh.model.occ.addPlaneSurface([loop])

gmsh.model.occ.synchronize()

# ---------------------------
# PART 2: Extrude fillet to create 3D volume
# ---------------------------
fillet_extrude = gmsh.model.occ.extrude([(2, fillet_surf)], 0, 0, W)
fillet_vol = fillet_extrude[1][1]

gmsh.model.occ.synchronize()

# ---------------------------
# PART 3: Create 2D arm cross-section at z=0
# ---------------------------
tx, ty, _ = tangent(a1)

arm_p1 = gmsh.model.occ.addPoint(Ro*math.cos(a1), Ro*math.sin(a1), 0)
arm_p2 = gmsh.model.occ.addPoint(Ro*math.cos(a1) + L_bar*tx, Ro*math.sin(a1) + L_bar*ty, 0)
arm_p3 = gmsh.model.occ.addPoint(Ri*math.cos(a1) + L_bar*tx, Ri*math.sin(a1) + L_bar*ty, 0)
arm_p4 = gmsh.model.occ.addPoint(Ri*math.cos(a1), Ri*math.sin(a1), 0)

arm_e1 = gmsh.model.occ.addLine(arm_p1, arm_p2)
arm_e2 = gmsh.model.occ.addLine(arm_p2, arm_p3)
arm_e3 = gmsh.model.occ.addLine(arm_p3, arm_p4)
arm_e4 = gmsh.model.occ.addLine(arm_p4, arm_p1)

arm_loop = gmsh.model.occ.addCurveLoop([arm_e1, arm_e2, arm_e3, arm_e4])
arm_surf = gmsh.model.occ.addPlaneSurface([arm_loop])

gmsh.model.occ.synchronize()

# ---------------------------
# PART 4: Extrude arm to create 3D volume
# ---------------------------
arm_extrude = gmsh.model.occ.extrude([(2, arm_surf)], 0, 0, W)
arm_vol = arm_extrude[1][1]

gmsh.model.occ.synchronize()

# ---------------------------
# PART 5: Fragment to merge the two volumes
# ---------------------------
gmsh.model.occ.fragment([(3, fillet_vol)], [(3, arm_vol)])
gmsh.model.occ.synchronize()

# ---------------------------
# PART 6: Rotate to final orientation
# ---------------------------
rot_angle = math.pi / 2 - a2
gmsh.model.occ.rotate(
    gmsh.model.getEntities(),
    0, 0, 0,
    0, 0, 1,
    rot_angle
)
gmsh.model.occ.synchronize()

# ---------------------------
# PART 7: Apply transfinite meshing after all boolean operations
# ---------------------------
# Transfinite constraints must be applied after fragment, as the operation
# regenerates all geometric entities.

all_curves = gmsh.model.getEntities(1)
all_surfaces = gmsh.model.getEntities(2)
all_volumes = gmsh.model.getEntities(3)

for dim, tag in all_curves:
    bbox = gmsh.model.getBoundingBox(dim, tag)
    x_min, y_min, z_min, x_max, y_max, z_max = bbox

    if abs(x_max - x_min) < 1e-6 and abs(y_max - y_min) < 1e-6:
        # Z-direction (thickness)
        gmsh.model.mesh.setTransfiniteCurve(tag, n_thickness)
    elif abs(z_max - z_min) < 1e-6:
        # XY-plane: classify by length
        curve_length = gmsh.model.occ.getMass(dim, tag)
        if curve_length < 0.015:
            gmsh.model.mesh.setTransfiniteCurve(tag, n_radial)
        elif curve_length < 0.040:
            gmsh.model.mesh.setTransfiniteCurve(tag, n_arc)
        else:
            gmsh.model.mesh.setTransfiniteCurve(tag, n_length)

for dim, tag in all_surfaces:
    boundaries = gmsh.model.getBoundary([(dim, tag)], oriented=False)
    if len(boundaries) == 4:
        try:
            gmsh.model.mesh.setTransfiniteSurface(tag)
            gmsh.model.mesh.setRecombine(2, tag)
        except Exception as e:
            print(f"Surface {tag}: could not apply transfinite - {e}")

for dim, tag in all_volumes:
    try:
        gmsh.model.mesh.setTransfiniteVolume(tag)
        gmsh.model.mesh.setRecombine(3, tag)
    except Exception as e:
        print(f"Volume {tag}: could not apply transfinite - {e}")

# ---------------------------
# PART 8: Physical groups for boundary conditions
# ---------------------------
vol_tags = [v[1] for v in all_volumes]
gmsh.model.addPhysicalGroup(3, vol_tags, tag=1)
gmsh.model.setPhysicalName(3, 1, "Volume")

dirichlet_surfaces = []
temperature_surfaces = []
midplane_surfaces = []

for surf_dim, surf_tag in all_surfaces:
    com = gmsh.model.occ.getCenterOfMass(surf_dim, surf_tag)
    x, y, z = com

    if abs(z) < 1e-6:
        midplane_surfaces.append(surf_tag)

    expected_x = (Ro + Ri)/2 * math.cos(a2)
    expected_y = (Ro + Ri)/2 * math.sin(a2)
    if abs(x - expected_x) < 1e-3 and abs(y - expected_y) < 1e-3:
        dirichlet_surfaces.append(surf_tag)

    arm_end_x = (Ro + Ri)/2 * math.cos(a1) + L_bar*tx
    arm_end_y = (Ro + Ri)/2 * math.sin(a1) + L_bar*ty
    if abs(x - arm_end_x) < 2e-3 and abs(y - arm_end_y) < 2e-3:
        temperature_surfaces.append(surf_tag)

if midplane_surfaces:
    gmsh.model.addPhysicalGroup(2, midplane_surfaces, tag=2)
    gmsh.model.setPhysicalName(2, 2, "Midplane_z0")

if dirichlet_surfaces:
    gmsh.model.addPhysicalGroup(2, dirichlet_surfaces, tag=3)
    gmsh.model.setPhysicalName(2, 3, "Dirichlet_Surface")

if temperature_surfaces:
    gmsh.model.addPhysicalGroup(2, temperature_surfaces, tag=4)
    gmsh.model.setPhysicalName(2, 4, "Temperature_Surface")

# ---------------------------
# PART 9: Generate and export mesh
# ---------------------------
gmsh.model.mesh.generate(3)
gmsh.write("mesh/domain_3D.msh")

gmsh.fltk.run()
gmsh.finalize()

# ---------------------------
# PART 10: Convert .msh to .xdmf via meshio
# ---------------------------
msh = meshio.read("mesh/domain_3D.msh")

cell_blocks = []
if hasattr(msh, "cells_dict") and msh.cells_dict:
    for ctype, data in msh.cells_dict.items():
        cell_blocks.append((ctype, data))
else:
    for block in msh.cells:
        if hasattr(block, "type") and hasattr(block, "data"):
            cell_blocks.append((block.type, block.data))
        else:
            ctype, data = block
            cell_blocks.append((ctype, data))

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
    print("Error: no 3D volumetric cell block found in mesh.")
    print("Available cell blocks:", [c for c, _ in cell_blocks])
    sys.exit(1)

cell_type, cells = selected

tags = None
if hasattr(msh, "cell_data_dict") and "gmsh:physical" in msh.cell_data_dict:
    phys = msh.cell_data_dict["gmsh:physical"]
    if isinstance(phys, dict):
        tags = phys.get(cell_type)
    elif isinstance(phys, list):
        for i, block in enumerate(getattr(msh, "cells", [])):
            btype = block.type if hasattr(block, "type") else block[0]
            if btype == cell_type:
                tags = phys[i] if i < len(phys) else None
                break

cell_data = {"gmsh:physical": [tags]} if tags is not None else None

out_mesh = meshio.Mesh(
    points=msh.points,
    cells=[(cell_type, cells)],
    cell_data=cell_data,
    field_data=getattr(msh, "field_data", None),
)

meshio.write("mesh/domain_3D.xdmf", out_mesh)
print(f"Converted domain_3D.msh -> domain_3D.xdmf ({cell_type}, {len(cells)} elements)")