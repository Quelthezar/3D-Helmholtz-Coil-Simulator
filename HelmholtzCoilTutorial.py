# import numpy as np
# import magpylib as magpy
# import pyvista as pv
#
# coil1 = magpy.Collection()
# for z in np.linspace(-8, 8, 16) * 1000:
#     winding = magpy.current.Circle(
#         current=100,
#         diameter=10,
#         position=(0,0,z),
#     )
#     coil1.add(winding)
#
# ts = np.linspace(-8, 8, 300)
# vertices = np.c_[5*np.cos(ts*2*np.pi), 5*np.sin(ts*2*np.pi), ts]
# coil2 = magpy.current.Polyline(
#     current=100,
#     vertices=vertices
# )
#
# # Create a finite sized Helmholtz coil-pair
# coil1 = magpy.Collection()
# for z in np.linspace(-1, 1, 5)*0.001:
#     for r in np.linspace(4, 5, 5) * 0.001:
#         winding = magpy.current.Circle(
#             current=10,
#             diameter=2*r,
#             position=(0,0,z),
#         )
#         coil1.add(winding)
#
# coil1.position = (0,0,5 * 0.001)
# coil2 = coil1.copy(position=(0,0,-5 * 0.001))
#
# helmholtz = magpy.Collection(coil1, coil2)
#
# # helmholtz.show()
#
# import matplotlib.pyplot as plt
# fig, ax = plt.subplots(1, 1, figsize=(6,5))
#
# # Compute field and plot the coil pair field on yz-grid
# grid = np.mgrid[0:0:1j, -13:13:20j, -13:13:20j].T[:,:,0]
# _, Y, Z = np.moveaxis(grid, 2, 0)
#
# B = magpy.getB(helmholtz, grid)
# _, By, Bz = np.moveaxis(B, 2, 0)
#
# Bamp = np.linalg.norm(B, axis=2)
# Bamp /= np.amax(Bamp)
#
# sp = ax.streamplot(Y, Z, By, Bz, density=2, color=Bamp,
#                    linewidth=np.sqrt(Bamp)*3, cmap='coolwarm',
#                    )
#
# # Plot coil outline
# from matplotlib.patches import Rectangle
# for loc in [(4,4), (4,-6), (-6,4), (-6,-6)]:
#     ax.add_patch(Rectangle(loc, 2, 2, color='k', zorder=10))
#
# # Figure styling
# ax.set(
#     title='Magnetic field of Helmholtz',
#     xlabel='y-position (m)',
#     ylabel='z-position (m)',
#     aspect=1,
# )
# plt.colorbar(sp.lines, ax=ax, label='(T)')
#
# plt.tight_layout()
# plt.show()
#
# fig, ax = plt.subplots(1, 1, figsize=(6,5))
#
# # Compute field of the coil pair on yz-grid
# grid = np.mgrid[0:0:1j, -3:3:20j, -3:3:20j].T[:,:,0]
# _, Y, Z = np.moveaxis(grid, 2, 0)
#
# B = helmholtz.getB(grid)
#
# # Field at center
# B0 = helmholtz.getB((0,0,0))
# B0amp = np.linalg.norm(B0)
#
# # Homogeneity error
# err = np.linalg.norm((B-B0)/B0amp, axis=2)
#
# # Plot error on grid
# sp = ax.contourf(Y, Z, err*100, levels=20)
#
# for loc in [(4,4), (4,-6), (-6,4), (-6,-6)]:
#     ax.add_patch(Rectangle(loc, 2, 2, color='k', zorder=10))
#
# # Figure styling
# ax.set(
#     title='Helmholtz homogeneity error',
#     xlabel='y-position (m)',
#     ylabel='z-position (m)',
#     aspect=1,
# )
# plt.colorbar(sp, ax=ax, label='(% of B0)')
#
# plt.tight_layout()
# plt.show()
#
# # Create a magnet with Magpylib
# magnet = magpy.magnet.Cylinder(polarization=(0, 0, 1), dimension=(0.010, 0.004))
#
# # Create a 3D grid with Pyvista
# grid = pv.ImageData(
#     dimensions=(41, 41, 41),
#     spacing=(0.001, 0.001, 0.001),
#     origin=(-0.02, -0.02, -0.02),
# )
#
# # Compute B-field and add as data to grid
# grid["B"] = helmholtz.getB(grid.points) * 1000  # T -> mT
#
# # Compute the field lines
# seed = pv.Disc(inner=0.001, outer=0.003, r_res=1, c_res=9)
# strl = grid.streamlines_from_source(
#     seed,
#     vectors="B",
#     max_step_length=0.1,
#     max_time=.02,
#     integration_direction="both",
# )
#
# # Create a Pyvista plotting scene
# pl = pv.Plotter()
#
# # Add magnet to scene - streamlines units are assumed to be meters
# magpy.show(helmholtz, canvas=pl, units_length="m", backend="pyvista")
#
# # Prepare legend parameters
# legend_args = {
#     "title": "B (mT)",
#     "title_font_size": 20,
#     "color": "black",
#     "position_y": 0.25,
#     "vertical": True,
# }
#
# # Add streamlines and legend to scene
# pl.add_mesh(
#     strl.tube(radius=0.0002),
#     cmap="bwr",
#     scalar_bar_args=legend_args,
# )
#
# # Prepare and show scene
# pl.camera.position = (0.03, 0.03, 0.03)
# pl.show()

import numpy as np
import magpylib as magpy
import pyvista as pv

# Create a finite sized Helmholtz coil-pair scaled to millimeters
coil1 = magpy.Collection()
for z in np.linspace(-1, 1, 5) * 0.001:  # Convert to millimeters
    for r in np.linspace(4, 5, 5) * 0.001:  # Convert to millimeters
        winding = magpy.current.Circle(
            current=10,
            diameter=2 * r,
            position=(0, 0, z),
        )
        coil1.add(winding)

coil1.position = (0, 0, 5 * 0.001)  # Convert position to millimeters
coil2 = coil1.copy(position=(0, 0, -5 * 0.001))

# Combine the coils into a Helmholtz coil configuration
helmholtz = magpy.Collection(coil1, coil2)

# Create a 3D grid with Pyvista, scaled to millimeter scale
grid = pv.ImageData(
    dimensions=(41, 41, 41),
    spacing=(0.001, 0.001, 0.001),  # Millimeter scale
    origin=(-0.02, -0.02, -0.02),   # Millimeter origin
)

# Compute the B-field of the Helmholtz coil and add it to the grid (scaled to millitesla)
grid["B"] = magpy.getB(helmholtz, grid.points) * 1000  # Convert Tesla to millitesla

# Compute the magnetic field lines (streamlines)
seed = pv.Disc(inner=0.001, outer=0.003, r_res=1, c_res=9)
strl = grid.streamlines_from_source(
    seed,
    vectors="B",
    max_step_length=0.1,
    max_time=.02,
    integration_direction="both",
)

# Create a Pyvista plotting scene
pl = pv.Plotter()

# Add Helmholtz coil to scene
magpy.show(helmholtz, canvas=pl, units_length="mm", backend="pyvista")

# Prepare the legend for the magnetic field strength
legend_args = {
    "title": "B (mT)",  # Display magnetic field strength in millitesla
    "title_font_size": 20,
    "color": "black",
    "position_y": 0.25,
    "vertical": True,
}

# Add streamlines and legend to the scene
pl.add_mesh(
    strl.tube(radius=0.0002),  # Streamline tubes
    cmap="bwr",  # Color map for the field lines
    scalar_bar_args=legend_args,
)

# Set up camera position for better visualization
pl.camera.position = (0.03, 0.03, 0.03)
pl.show()
