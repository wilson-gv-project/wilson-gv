from typing import Any

# State, energy, displacement
class VibState:
	"""
	Class to represent a vibrational state.
	This is for a "concrete" vibrational state and not the same as its symbolic namesake in wilson-derive.
	"""

	def __init__(self, s: dict, e: float, d: Any=None):
		"""
		s: dictionary {(harm. quanta): coeff, (harm. quanta): coeff, ...}: Specify the state in terms of harm. osc. WFs
		e: float: State energy level
		d: type not specified: Should be some form of vector to represent displacement in terms of atomic coordinates.
		"""

		self.s = s
		self.e = e
		self.d = d

	def __repr__(self):
		return f"vibState {self.s}, energy is {self.e} cm-1"


