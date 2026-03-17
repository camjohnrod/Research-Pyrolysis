import os
import sys
import gmsh
import meshio
import numpy as np


# ================================================================
# Convert .msh to .xdmf using meshio
# ================================================================

filename_old = "meso_rve_3D.msh"
filename_new = "meso_rve_3D.xdmf"

# Read mesh
msh = meshio.read("mesh/meso_rve_3D.msh")

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

# msh.points += np.array([0.5, 0.5, 0.01])

out_mesh = meshio.Mesh(
    points=msh.points,
    cells=[(cell_type, cells)],
    cell_data=cell_data,
    field_data=getattr(msh, "field_data", None),
)

meshio.write(f"mesh/{filename_new}", out_mesh)
print(f"Finished converting {filename_new} from {filename_old}")