import matplotlib.pyplot as plt

from .prop_trivname import prop_trivname


def get_plt_latex(expr, savename: str | None = None, color: str='black'):
    """
    color: blue, green, red, black, yellow ...
    """
    _, ax = plt.subplots(figsize=(22, 2))
    ax.axis("off")  # hide axes

    ax.text(
        0.5, 0.5, f"${expr}$",
        size=50,
        color=color,
        ha="center",
        va="center",
        math_fontfamily="cm"  # use Computer Modern
    )

    if savename is None:
        plt.tight_layout()
        plt.show()
    else:
        plt.savefig(savename)

def prop_trivialname_latex(geo=None, el=None, rot=None) -> tuple[str, str]:
	"""
	Generate an abbreviated trivial name for a molecular property based on orders of perturbation

	ord_geo: Order of geometric differentiation (perturbation)
	ord_el: Order of electric dipole perturbation
	ord_rot: (Ad hoc) order of rotational interaction: Used for rotational constants and Coriolis coupling constants
	"""
	from ..wilson_utils.common_labels import op_labels_greek_latex
	if geo is None:
		geo = []
	if el is None:
		el = []
	if rot is None:
		rot = []
	
	ord_geo = len(geo)
	ord_el = len(el)

	# prop_trivname raises for orders without a name
	name = prop_trivname(ord_geo=ord_geo, ord_el=ord_el, ord_rot=len(rot))
	geo_inds = ''.join([rf'\partial Q_{{{ind}}}' for ind in geo])

	if name in ('E', 'B', 'coriolis'):
		return name, geo_inds

	triv_el = {0: 'E', 1: r'\mu', 2: r'\alpha', 3: r'\beta', 4: r'\gamma', 5: r'\chi^{(3)}', 6: r'\chi^{(4)}'}
	triv_geo = {0: '', 1: r'\partial', 2: r'\partial^{2}', 3: r'\partial^{3}', 4: r'\partial^{4}', 5: r'\partial^{5}', 6: r'\partial^{6}'}
	el_ops_greek_inds = ''.join([op_labels_greek_latex[i] for i in el])

	return triv_geo[ord_geo] + triv_el[ord_el] + rf'_{{{el_ops_greek_inds}}}', geo_inds