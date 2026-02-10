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
theta = math.radians(54.0)
L = 60.0e-3       # length of the rectangular arm
W = 110.0e-3      # thickness in z

# Transfinite mesh divisions
n_radial = 5      # divisions along radial direction (inner to outer)
n_arc = 13        # divisions along arc
n_length = 19     # divisions along arm length
n_thickness = 5   # divisions through thickness (z-direction)

half = theta / 2
a1 = math.pi/2 - half  # left end of arc (where arm will extend)
a2 = math.pi/2 + half  # right end of arc

def tangent(a):
    return math.sin(a), -math.cos(a), 0.0

# ============================================================================
# ISSUE IDENTIFIED: The fragment operation destroys transfinite constraints!
# SOLUTION: Apply transfinite meshing AFTER fragment operation to the 
# resulting merged geometry
# ============================================================================

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

# DON'T apply transfinite here - it will be lost after fragment operation

# ---------------------------
# PART 2: Extrude fillet to create 3D volume (without transfinite yet)
# ---------------------------
fillet_extrude = gmsh.model.occ.extrude(
    [(2, fillet_surf)], 
    0, 0, W
)

fillet_vol = fillet_extrude[1][1]

gmsh.model.occ.synchronize()

# ---------------------------
# PART 3: Create 2D arm cross-section at z=0
# ---------------------------
tx, ty, _ = tangent(a1)

arm_p1 = gmsh.model.occ.addPoint(Ro*math.cos(a1), Ro*math.sin(a1), 0)
arm_p2 = gmsh.model.occ.addPoint(Ro*math.cos(a1) + L*tx, Ro*math.sin(a1) + L*ty, 0)
arm_p3 = gmsh.model.occ.addPoint(Ri*math.cos(a1) + L*tx, Ri*math.sin(a1) + L*ty, 0)
arm_p4 = gmsh.model.occ.addPoint(Ri*math.cos(a1), Ri*math.sin(a1), 0)

arm_e1 = gmsh.model.occ.addLine(arm_p1, arm_p2)
arm_e2 = gmsh.model.occ.addLine(arm_p2, arm_p3)
arm_e3 = gmsh.model.occ.addLine(arm_p3, arm_p4)
arm_e4 = gmsh.model.occ.addLine(arm_p4, arm_p1)

arm_loop = gmsh.model.occ.addCurveLoop([arm_e1, arm_e2, arm_e3, arm_e4])
arm_surf = gmsh.model.occ.addPlaneSurface([arm_loop])

gmsh.model.occ.synchronize()

# DON'T apply transfinite here - it will be lost after fragment operation

# ---------------------------
# PART 4: Extrude arm to create 3D volume (without transfinite yet)
# ---------------------------
arm_extrude = gmsh.model.occ.extrude(
    [(2, arm_surf)], 
    0, 0, W
)

arm_vol = arm_extrude[1][1]

gmsh.model.occ.synchronize()

# ---------------------------
# PART 5: Fragment to merge the two volumes
# ---------------------------
out_dimtags, out_map = gmsh.model.occ.fragment([(3, fillet_vol)], [(3, arm_vol)])

gmsh.model.occ.synchronize()

# ---------------------------
# PART 6: Translate everything so z=0 is at the midplane
# ---------------------------
all_entities = gmsh.model.getEntities()
gmsh.model.occ.translate(all_entities, 0, 0, -W/2)

gmsh.model.occ.synchronize()

# ============================================================================
# CRITICAL FIX: Apply transfinite meshing AFTER all boolean operations
# ============================================================================

print("\n=== Applying Transfinite Mesh Constraints ===")

# Get all curves and surfaces after fragment operation
all_curves = gmsh.model.getEntities(1)
all_surfaces = gmsh.model.getEntities(2)

# Strategy: Classify curves by their geometric properties and apply 
# appropriate transfinite divisions

for dim, tag in all_curves:
    # Get curve bounding box and length
    bbox = gmsh.model.getBoundingBox(dim, tag)
    x_min, y_min, z_min, x_max, y_max, z_max = bbox
    
    # Curves in z-direction (thickness)
    if abs(x_max - x_min) < 1e-6 and abs(y_max - y_min) < 1e-6:
        gmsh.model.mesh.setTransfiniteCurve(tag, n_thickness)
        print(f"Curve {tag}: z-direction -> {n_thickness} divisions")
    
    # Curves in xy-plane (radial direction: short curves)
    elif abs(z_max - z_min) < 1e-6:
        # Calculate curve length to distinguish radial vs arc vs arm length
        mass_data = gmsh.model.occ.getMass(dim, tag)
        curve_length = mass_data
        
        # Radial curves (thickness of the beam: Ro-Ri ≈ 7mm)
        if curve_length < 0.015:  # Less than 15mm
            gmsh.model.mesh.setTransfiniteCurve(tag, n_radial)
            print(f"Curve {tag}: radial (L={curve_length*1000:.1f}mm) -> {n_radial} divisions")
        
        # Arc curves (around the fillet)
        elif curve_length < 0.040:  # Between 15-40mm (arc portion)
            gmsh.model.mesh.setTransfiniteCurve(tag, n_arc)
            print(f"Curve {tag}: arc (L={curve_length*1000:.1f}mm) -> {n_arc} divisions")
        
        # Arm length curves
        else:  # Longer curves (arm length ≈ 60mm)
            gmsh.model.mesh.setTransfiniteCurve(tag, n_length)
            print(f"Curve {tag}: arm length (L={curve_length*1000:.1f}mm) -> {n_length} divisions")

# Apply transfinite to all quadrilateral surfaces and recombine
print("\n=== Applying Transfinite Surfaces ===")
for dim, tag in all_surfaces:
    # Get the surface's bounding curves
    boundaries = gmsh.model.getBoundary([(dim, tag)], oriented=False)
    
    # Only apply to surfaces with 4 boundaries (quadrilateral topology)
    if len(boundaries) == 4:
        try:
            gmsh.model.mesh.setTransfiniteSurface(tag)
            gmsh.model.mesh.setRecombine(2, tag)
            print(f"Surface {tag}: transfinite + recombine applied")
        except Exception as e:
            print(f"Surface {tag}: Could not apply transfinite - {e}")

# Apply transfinite to volumes
print("\n=== Applying Transfinite Volumes ===")
all_volumes = gmsh.model.getEntities(3)
for dim, tag in all_volumes:
    try:
        gmsh.model.mesh.setTransfiniteVolume(tag)
        gmsh.model.mesh.setRecombine(3, tag)
        print(f"Volume {tag}: transfinite + recombine applied")
    except Exception as e:
        print(f"Volume {tag}: Could not apply transfinite - {e}")

# ---------------------------
# Physical Groups for Boundary Conditions
# ---------------------------

# Create a physical group for the entire volume
vol_tags = [v[1] for v in all_volumes]
gmsh.model.addPhysicalGroup(3, vol_tags, tag=1)
gmsh.model.setPhysicalName(3, 1, "Volume")

# Find surfaces by their center of mass
dirichlet_surfaces = []
temperature_surfaces = []
midplane_surfaces = []

for surf_dim, surf_tag in all_surfaces:
    com = gmsh.model.occ.getCenterOfMass(surf_dim, surf_tag)
    x, y, z = com
    
    # Midplane at z=0 (after translation)
    if abs(z) < 1e-6:
        midplane_surfaces.append(surf_tag)
    
    # Right radial surface at angle a2
    expected_x = (Ro + Ri)/2 * math.cos(a2)
    expected_y = (Ro + Ri)/2 * math.sin(a2)
    
    if abs(x - expected_x) < 1e-3 and abs(y - expected_y) < 1e-3:
        dirichlet_surfaces.append(surf_tag)
    
    # Arm end surface
    arm_end_x = (Ro + Ri)/2 * math.cos(a1) + L*tx
    arm_end_y = (Ro + Ri)/2 * math.sin(a1) + L*ty
    
    if abs(x - arm_end_x) < 2e-3 and abs(y - arm_end_y) < 2e-3:
        temperature_surfaces.append(surf_tag)

# Create physical groups
if midplane_surfaces:
    gmsh.model.addPhysicalGroup(2, midplane_surfaces, tag=2)
    gmsh.model.setPhysicalName(2, 2, "Midplane_z0")
    print(f"\nTagged {len(midplane_surfaces)} midplane surface(s) at z=0")

if dirichlet_surfaces:
    gmsh.model.addPhysicalGroup(2, dirichlet_surfaces, tag=3)
    gmsh.model.setPhysicalName(2, 3, "Dirichlet_Surface")
    print(f"Tagged {len(dirichlet_surfaces)} Dirichlet surface(s)")

if temperature_surfaces:
    gmsh.model.addPhysicalGroup(2, temperature_surfaces, tag=4)
    gmsh.model.setPhysicalName(2, 4, "Temperature_Surface")
    print(f"Tagged {len(temperature_surfaces)} Temperature surface(s)")

# ---------------------------
# Mesh generation
# ---------------------------
print("\n=== Generating Mesh ===")
gmsh.model.mesh.generate(3)
gmsh.write("mesh/domain_3D.msh")
print("Mesh written to mesh/domain_3D.msh")

# Optional: launch GUI to inspect mesh
gmsh.fltk.run()

gmsh.finalize()

# ================================================================
# Convert .msh to .xdmf using meshio
# ================================================================

print("\n=== Converting to XDMF ===")

filename_old = "domain_3D.msh"
filename_new = "domain_3D.xdmf"

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
    print("Error: no 3D volumetric cell block (tetra/hexahedron/wedge/pyramid/prism) found in mesh.")
    print("Available cell blocks:", [c for c, _ in cell_blocks])
    sys.exit(1)

cell_type, cells = selected
print(f"Found cell type: {cell_type} with {len(cells)} elements")

tags = None
if hasattr(msh, "cell_data_dict") and "gmsh:physical" in msh.cell_data_dict:
    phys = msh.cell_data_dict["gmsh:physical"]
    if isinstance(phys, dict):
        tags = phys.get(cell_type)
    elif isinstance(phys, list):
        idx = None
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
print(f"\nGeometry: L-beam in x-y plane at z=0, thickness from z={-W/2:.4f} to z={W/2:.4f}")
print(f"Midplane (symmetry plane) tagged as 'Midplane_z0' (tag=2)")