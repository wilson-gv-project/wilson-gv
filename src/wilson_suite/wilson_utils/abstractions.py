from dataclasses import dataclass, field, asdict, is_dataclass, InitVar
from typing import Any

@dataclass
class VibState:
	"""
	Class to represent a vibrational state.
	This is for a "concrete" vibrational state and not the same as its symbolic namesake in wilson-derive.

	----
	s: dictionary {(harm. quanta): coeff, (harm. quanta): coeff, ...}: Specify the state in terms of harm. osc. WFs
	e: float: State energy level
	d: type not specified: Should be some form of vector to represent displacement in terms of atomic coordinates
	
	UPD:
	dictionary self.s is not JSON-serializable (tuples can't be keys), but self.serial_s is.
	self.serial_s is set up in post_init; deserialize_state_dict will return original self.s based on self.serial_s.

	Notes:
	s: InitVar[dict] = field(repr=False) - means that this atribute will not be in repr() of the class instance
	InitVar - is an init-only variable
	This seems to be okay for now, but should mind this feature
	"""
	s: InitVar[dict] = field(repr=False) # 
	e: float = 0.0
	d: Any = None
	serial_s: dict = field(init=False)

	def __post_init__(self, s):
		self.serial_s = {",".join(k): v for k, v in s.items()}

	# def __post_init__(self):
		# self.serial_s = {",".join(k): v for k, v in self.s.items()}

	def deserialize_state_dict(self) -> dict:
		return {tuple(k.split(",")): v for k, v in self.serial_s.items()}

