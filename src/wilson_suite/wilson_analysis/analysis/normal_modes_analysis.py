"""
https://kthpanor.github.io/echem/docs/visualize/opt_vib_vis.html
"""
import py3Dmol as p3d
import numpy as np

from scipy.constants import physical_constants
bohr_in_angstroms = physical_constants['Bohr radius'][0] / 10 ** (-10)

# To animate the normal mode we will need both the geometry and the displacements
def make_normal_mode_str(elements: list, coords: np.ndarray, normal_mode: np.ndarray):
    """
    molecule = (labels, coordinatesA, natoms)

    normal_mode
    """
    natm = len(elements)
    vib_xyz = "%d\n\n" % natm
    nm = normal_mode.reshape(natm, 3)
    for i in range(natm):
        # add coordinates:
        vib_xyz += elements[i] + " %15.7f %15.7f %15.7f " % (coords[i, 0], coords[i, 1], coords[i, 2])
        # add displacements:
        vib_xyz += "%15.7f %15.7f %15.7f\n" % (nm[i, 0], nm[i, 1], nm[i, 2])
    return vib_xyz

def show_mode(choice: int, elements: list, coords: np.ndarray, normal_mode_dict: dict, toA=False):
    """
    show_mode(9, gParser)
    show_mode(7, c4Parser, True)
    """
    if toA:
        f = physical_constants['Bohr radius'][0] / 10 ** (-10)
    else:
        f = 1.
    normal_mode = make_normal_mode_str(elements, coords * f, normal_mode_dict[choice])
    view = p3d.view(viewergrid=(1, 1), width=600, height=300)
    view.addModel(normal_mode, "xyz", {'vibrate': {'frames': 10, 'amplitude': 0.75}})

    view.setViewStyle({"style": "outline", "width": 0.05})
    view.setStyle({"stick": {}, "sphere": {"scale": 0.25}})
    
    view.rotate(90, {'x':0, 'y':1, 'z':0}) # Rotates 90 degrees around the Y-axis
    
    view.animate({'loop': 'backAndForth'})
    view.zoomTo()
    view.show()

