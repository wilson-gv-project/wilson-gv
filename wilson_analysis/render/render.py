from .matplotlib_renderer import MatplotlibRenderer

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from wilson_main.abstractions import SimContext

def render_spectrum(context: 'SimContext') -> None:
    
    """
    High-level function to render spectrum with specified backend
 -------------
    spec_data is not the final Z values of spectrum (intensities)
        it is amplitudes here, complex values, but they get to be transformed
        using spec_data_operations attribute of RenderingInfo instance

    context = {'spec': self.spec, 'system': self.system, 
                'exp': self.exp, 'diagn': self.diagn, 
                'name': self.name, 
                'spec_eval_setup': self.spec_eval_setup,
                'do_diagn': True}
--------------

    what are the options for rendering? 
    
    by plot type/projection type
    1. 1D plot : slice of a 2D spectrum; 1D IR/Raman spectrum (?)
    2. 2D plot : 2D IR/Raman spectrum; slice of a 3D spectrum : a) contour plot; b) scatter plot
    3. 3D plot : 2D IR/Raman spectrum : surface plot == contout plot (same info; Z axis is intensity or color is intensity)

    by spectrum dimensionality
    - 1D spectrum: 1D IR/Raman spectrum - simply 1d plot
    - 2D spectrum: 2D IR/Raman spectrum - 2D contour/scatter plot or 3D surface plot
    - 3D spectrum: 3D IR/Raman spectrum - 1) 2D slice of a 3D spectrum - 2D contour/scatter plot or 3D surface plot; 2) 3D surface plot with color as intensity
    - nD spectrum: nD IR/Raman spectrum - lower D slices as above
    
    """
    spec_data = context.spec
    spec_eval_setup = context.spec_eval_setup
    filename = context.filename
    backend = context.backend

    # TODO not used currently
    do_diagn = context.do_diagn
    diagn = {}

    if backend == 'matplotlib':
        renderer_class=MatplotlibRenderer
    else:
        raise NotImplementedError('Only matplotlib backend is currently supported')
    
    plot_config = spec_eval_setup.rnd_info.style_config

    renderer = renderer_class(spec_data=spec_data, 
                              spec_grid=spec_eval_setup.grid,
                              ev_info=spec_eval_setup.ev_info, 
                              rnd_info=spec_eval_setup.rnd_info, 
                              config=plot_config, do_diagn=do_diagn)
    fig, ax, contour, cbar = renderer.render(filename)
    
    if do_diagn:
        return tuple([fig, ax, contour, cbar]), diagn
    else:
        return tuple([fig, ax, contour, cbar])

