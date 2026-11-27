from wilson_suite.wilson_analysis.render.render_utils import NormalizationType, PlotConfig
import pytest


def test_NormalizationType():
    print('\nNormalizationType.LOG_SCALE', NormalizationType.LOG_SCALE, '\n')
    for normtype in NormalizationType:
        print(normtype)

    print('\n', list(NormalizationType))

    # Convert string to Enum
    method_name = "log_ratio"
    method = NormalizationType(method_name)
    print(method)

    with pytest.raises(ValueError) as excinfo:
        method_name = "smth"
        method = NormalizationType(method_name)
        print(method)
    assert 'is not a valid NormalizationType' in str(excinfo.value)


def test_PlotConfig():
    assert PlotConfig() == PlotConfig(figsize=(35,45),
                                      label_fontsize=25,
                                      font_dict={'size': 20},
                                      colormap='magma',
                                      saturation_color='#FF00FF',
                                      dpi=250,
                                      tick_step=200,
                                      equal_aspect=True,
                                      other_colors=True,
                                      no_data_color='#E0E0E0',
                                      below_range_color='#F8F8F8',
                                      data_edge_color='black',
                                      data_edge_width=0.75,
                                      x_min=None,x_max=None,y_min=None,y_max=None,
                                      colorbar_main_label='Intensity',
                                      colorbar_padding=0.01,
                                      show_top_ticks=False,
                                      show_right_ticks=False,
                                      x_tick_rotation=45,
                                      colormap_spacing='log')
