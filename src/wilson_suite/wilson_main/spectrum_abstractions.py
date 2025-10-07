from dataclasses import dataclass, field
from typing import Any
import numpy as np
from ..wilson_analysis.render.spectrum_renderer import PlotConfig, NormalizationType

import logging
logger = logging.getLogger("wilson")

@dataclass
class SpectralAxis:
	"""
	'Plain' spectral axis for rendering response function freq arg spectra;
	with independent lineshape functions.

	freq_vars is {freq label 1 in this axis: coeff, ...}

	Examples:
		axis1 = ws.main.abstractions.spectralAxis({1: 1})        -- w1
		axis2 = ws.main.abstractions.spectralAxis({1: 1, 2: -1}) -- w1-w2

	simple w1 and w2:
		axis1 = ws.main.abstractions.spectralAxis({1: 1})        -- w1
		axis2 = ws.main.abstractions.spectralAxis({2: 1})        -- w2
	"""
	freq_vars: dict
	
	def __post_init__(self):
		if not isinstance(self.freq_vars, dict):
			raise TypeError('SpectralAxis needs freq_vars to be a dictionary like {freq label 1 in this axis: coeff, ...}')


# simply copying old sketch for now
class SpectralAxisAdvanced:
	"""
	Class to represent an "advanced" spectral axis (involving e.g. variation of experiment parameters
	or possibly other attributes). Not yet implemented.
	"""

	def __init__(self):

		pass

# Fixme: Have this be just SpectralGrid to replace
@dataclass
class SpectralGridMultires:

    base_resolution: float
    base_axis_npoints: list
    axis_starts: list
    increased_res_regions: list # One entry per region,
    # region def: [[axis starts], [n points wrt base resolution], detail level (integer or power or 2)]

    # Return all grid points as tuples
    def yield_all_grid_points(self):

        pass

    # Other yield methods? Discuss needs

def is_axis_cfg_valid(axis_cfg, valid_cfgs):

    # TODO: Make canonical sorting before checking if in
    if not axis_cfg in valid_cfgs:
        return False

    return True

# FIXME: RM for now
# Holds an axis definer and a grid
# Comment "Frame" suggests 2D but not actually limited to that
# FIXME: Settle organization of axis check, (default) use of canonical axes, (default?) grid choices
@dataclass
class SpectralFrame:

    axes: dict # dict of SpectralAxis instances
    grid: SpectralGridMultires

    def __post_init__(self, valid_axes):

        for i in self.axes:
            if not isinstance(self.axes[i], SpectralAxis):
                raise TypeError("Values of axes dict should be SpectralAxis instances")

        if not is_axis_cfg_valid(self.axes, valid_axes):
            raise AssertionError("Axis choice does not correspond to valid set")




@dataclass
class SpectralGrid:
	"""
	Class to represent a collective set of spectral axes.

	Use example:

	axis1 = ws.main.abstractions.spectralAxis({1: 1})
	axis2 = ws.main.abstractions.spectralAxis({1: 1, 2: -1})
	start = {1: 250, 2: 100}
	end = {1: 3850, 2: 7550}
	spacer = {1: 3.8, 2: 3.8}
	spec_grid = ws.main.abstractions.spectralGrid({1: axis1, 2: axis2}, range_style='uniform',
												  start=start, end=end, spacer=spacer)
	
	----
	axes: Dictionary {axis 1 ID: SpectralAxis instance, axis 2 ID: SpectralAxis instance, ...}: One SpectralAxis
	instance per axis. TODO: Also to support instances being SpectralAxisAdvanced
		this parameter is misleading if :
			e.g., axis1 is with {1: 1} and axis2 is with {1: 1, 2: -1} 
			and `spacer`, `start` and `end` dicts are used as they are now in __post_init__ and `make_mesh_numpy`
			the grid itself right now would not correspond to axis1 is with {1: 1} and axis2 is with {1: 1, 2: -1}

	range_style: String: What sort of range? Intended options at least "uniform" or "custom"
	start: Dictionary {axis 1 ID: starting point (float), ...}: Axis starting points
	end: Dictionary {axis 1 ID: end point (float), ...}: Axis end points
	n_pts: Dictionary {axis 1 ID: number of points (int), ...}: Number of points by axis
	spacer: Dictionary {axis 1 ID: spacer (float), ...}: Spacers by axis
	custom_range: Type not specified: Custom range for each axis. Not yet implemented
	collective_grid: Type not specified (but most likely will be ndarray): (Custom) collective grid for all axes.
	"""
	axes: dict
	range_style: str
	start: dict=None
	end: dict=None
	n_pts: dict=None
	spacer: dict=None
	custom_range: dict=None
	# Optional collective (e.g. adaptive) grid
	# Otherwise intended to default to full granularity grid of individual axes
	collective_grid: Any=None

	def __post_init__(self):
		
		for i in self.axes:
			if not isinstance(self.axes[i], SpectralAxis):
				raise TypeError("Values of axes dict should be SpectralAxis instances")

		if (self.range_style == 'uniform'):

			self.ranges = {}
			n_pts = {}
			spacer = {}

			for i in self.axes:

				if (self.n_pts is None) and (self.spacer is None):
					raise AssertionError('For a uniform setup, either a spacer or a n_pts dictionary must be specified')

				if (self.n_pts is not None) and (self.spacer is not None):
					raise AssertionError('Only one of the arguments n_pts and spacer may be specified')

				if self.n_pts is not None:

					spacer[i] = (self.end[i] - self.start[i])/(self.n_pts[i] + 1)

				elif self.spacer is not None:

					# Underflow possible
					n_pts[i] = int((self.end[i] - self.start[i])/self.spacer[i] + 1)
					if not(self.end[i] == self.start[i] + self.spacer[i]*(n_pts[i] - 1)):
						logger.warning(f'NOTE: Axis defined end {self.end[i]} not precisely at spacer increment of start')

				else:

					raise AssertionError('For uniform grid, must specify either spacer or n_pts')

				# fixme: Other datatype? Should be fine for now
				self.ranges[i] = np.arange(self.start[i], self.end[i], self.spacer[i])
			
			if spacer:
				self.spacer = spacer
			if n_pts:
				self.n_pts = n_pts

		if(self.range_style == 'custom'):
			# rm error to enable skipping `spacer, start, end` - they aren't used meaningfully anyway
			logger.warning('Custom range style is not yet supported')
			pass


	def make_mesh_numpy(self) -> dict:
		"""
		Make a meshgrid using the axes information
		"""

		listofmeshaxes = []
		for ax_label in self.axes:
			if self.spacer is not None:
				wn = np.arange(self.start[ax_label], self.end[ax_label], self.spacer[ax_label])
				listofmeshaxes.append(wn)
			elif self.n_pts is not None:
				wn = np.linspace(self.start[ax_label], self.end[ax_label], self.n_pts[ax_label])
				listofmeshaxes.append(wn)
		meshes = np.meshgrid(*listofmeshaxes)

		mesh_dict = {}
		for i, ax_label in enumerate(self.axes):
			mesh_dict[ax_label] = meshes[i]

		return mesh_dict

	def collGridFromAxes(self):
		"""
		Make collective grid from individual axes linspaces. Not yet implemented
		"""

		pass


@dataclass
class EvaluationVariable:
	"""
	Like SpectralAxis, but a range for an independent variable of the response function (frequency variable)

	range_style: 'uniform' or 'custom'
	"""
	range_style: str
	start: float = None
	end: float = None
	n_pts: int = None
	spacer: float = None
	custom_range: list|np.ndarray = None

	def __post_init__(self):
		"""
		dealing with one range at the time seems to be more clean
		"""
		
		if self.range_style == 'custom':
			raise NotImplementedError('Custom range style is not yet supported')
		
		elif self.range_style == 'uniform':
			
			if (self.n_pts is None) and (self.spacer is None):
				raise AssertionError('For a uniform setup, either a spacer or a n_pts dictionary must be specified')

			if (self.n_pts is not None) and (self.spacer is not None):
				raise AssertionError('Only one of the arguments n_pts and spacer may be specified')
			
			if self.n_pts is not None:
				self.spacer = (self.end - self.start)/(self.n_pts + 1)
				self.range = np.linspace(self.start, self.end, self.n_pts)

			elif self.spacer is not None:

				self.n_pts = int((self.end - self.start)/self.spacer + 1)
				if self.end != self.start + self.spacer*(self.n_pts - 1):
					logger.info(f'NOTE: Axis defined end {self.end} not precisely at spacer increment of start')
				self.range = np.arange(self.start, self.end, self.spacer)


@dataclass
class EvaluationInfo:
	"""
	this feels a bit more "official" than a dict
	and it is warranted because that is a critical info that is needed for the evaluation

	freq_variables - is a dict {variable label: variable data} with a range for each
	fixed_variables - a dict of values for the non-varied fixed variables 
		(e.g., when having a 2D slice of a 3D spectrum at fixed 3rd)
	"""
	freq_variables: dict
	Gamma: float
	Gamma_unit: str
	freq_condition: str = None
	fixed_variables: dict = field(default_factory=lambda: dict())
	# 'diag_margin'- this parameter is specific to the condition ow w2>w1
	spec_result: np.ndarray | dict = None
	margins: dict = None

	@property
	def spec_window_bounds(self):
		"""
		creating `bounds` dict for `check_if_in_window()`
		"""
		bounds = {}
		for key in self.freq_variables:
			bounds[key] = {'left': np.min(self.freq_variables[key]) + self.margins.get(key, 0.), 
						'right': np.max(self.freq_variables[key]) + self.margins.get(key, 0.)}

		return bounds

@dataclass
class RenderingInfo:
	"""
	this feels a bit more "official" than a dict
	and it is warranted because that is a critical info that is needed for the rendiring

	projection: '1d', '2d' or '3d'
	reference_max: normalizing to this reference_max value

	"""
	projection: str = '2d'
	reference_max: float = None
	dynamic_range: float = 100
	num_levels: int = 12
	intensity_normalization_type: NormalizationType = NormalizationType.LOG_SCALE
	title: str = 'plot'
	spec_data_operations: str = 'abs()**2'  # 'abs', 'real', 'imag', 'abs()**2'
	metadata: dict = field(default_factory=lambda: dict())
	to_save: bool = False
	filename: str = 'spectrum.svg'
	backend: str = 'matplotlib'
	# style configurations - currently will work/be used for matplotlib renderer
	style_config: PlotConfig = field(default_factory=lambda: PlotConfig())

# An evaluation setup contains various visualization configuration information
# and information about other relevant evaluation-related choices for a wilsonSimulation instance
#
# Examples of relevant information here:
# Evaluation grid
# System to run simulation on
@dataclass
class SpecEvalSetup:
	"""
	Class for setup information related to spectrum evaluation and rendering
	FIXME: Consider making this into a dataclass
	
	----
	grid: SpectralGrid instance: The grid on which the spectrum is to be evaluated
	ev_info: dict: Setup information which is principally evaluation-related (e.g. dynamic range, relaxation
	parameters etc.)
	rnd_info: dict: Setup information which is principally rendering-related (e.g. number of level ticks, other
	plotting-/visualization-related information)
	FIXME: Consider formalizing which setup attributes may be passed in ev_info and rnd_info -> RenderingInfom and EvaluationInfo
	"""
	grid: SpectralGrid=None
	ev_info: EvaluationInfo = None
	rnd_info: RenderingInfo = None

	def __post_init__(self):
		if self.grid is not None:
			if not isinstance(self.grid, SpectralGrid):
				raise TypeError("Values of axes dict should be SpectralAxis instances")

