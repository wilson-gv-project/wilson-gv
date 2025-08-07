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

