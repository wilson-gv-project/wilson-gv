import numpy as np
import pandas as pd
import altair as alt
import math


class DotSpectrum:

    def __init__(self, locations_dict, intensities_dict):

        self.locations_dict = locations_dict # (a, b): (w1, w2)
        self.intensities_dict = intensities_dict # (a, b): intensity


    def plot_va(self, w=1100, h=700):
        data = {
            'ab': [(int(i[0]), int(i[1])) for i in list(self.intensities_dict.keys())],
            'intensity': [float(i) for i in list(self.intensities_dict.values())],
            'log10(Intensity)': np.log10(np.array(list(self.intensities_dict.values()))),
            'w1': [self.locations_dict[k][0] for k in self.intensities_dict.keys()],
            'w2': [self.locations_dict[k][1] for k in self.intensities_dict.keys()],
            'w2-w1': [self.locations_dict[k][1]-self.locations_dict[k][0] for k in self.intensities_dict.keys()]
        }

        data = {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in data.items()}
        df = pd.DataFrame(data)

        #
        # threshold_slider = alt.binding_range(min=math.floor(min(df['log10(Intensity)'])) - 1,
        #                                      max=math.ceil(max(df['log10(Intensity)'])) + 1,
        #                                      step=0.1, name='Threshold:')
        # threshold_select = alt.selection_point(fields=['log10(Intensity)'], bind=threshold_slider)
        # pd.set_option('display.max_rows', 500)
        # pd.set_option('display.max_columns', 500)
        # pd.set_option('display.width', 1000)
        #
        # # title = alt.TitleParams(f'Terms: {tuple(self.selection)}', anchor='middle')
        # chart = alt.Chart(df).mark_circle().encode(
        #     x='omega1',
        #     y='w2mw1',
        #     color=alt.condition(
        #         alt.datum['log10(Intensity)'] > alt.expr.if_(threshold_select, threshold_select['log10(Intensity)'],
        #                                                      0),
        #         alt.value('steelblue'),  # Color for points above the threshold
        #         alt.value('lightgray')  # Color for points below the threshold
        #     ),
        #     tooltip=[alt.Tooltip('omega1', format='.2f'),
        #              alt.Tooltip('omega2', format='.2f'),
        #              alt.Tooltip('w2mw1', format='.2f'),
        #              alt.Tooltip('log10(Intensity)', format='.4f'),
        #              alt.Tooltip('Intensity', format='.4e'),
        #              # alt.Tooltip('relative el/mech', format='.5f'),
        #              'a', 'b', 'type'
        #              ],
        #     opacity=alt.condition(
        #         alt.datum['log10(Intensity)'] > alt.expr.if_(threshold_select, threshold_select['log10(Intensity)'],
        #                                                      0),
        #         alt.value(1),  # Full opacity for points above the threshold
        #         alt.value(0.35)  # No opacity for points below the threshold
        #     )
        # ).add_selection(
        #     threshold_select
        # ).properties(
        #     width=w,
        #     height=h
        # ).interactive()
        #
        # # alt.renderers.enable("browser")
        # alt.renderers.enable("jupyterlab")
        # # chart.save(prefix+'_resints.html', inline=True, scale_factor=2)

        # return chart
        return df

class DotSpecStorage:

    def __init__(self):
        self.terms = {}  # Dictionary: term_id -> Term2D object

    def add_term_dotspec(self, term):
        if term.term_id in self.terms:
            print(f"Warning: Overwriting existing term {term.term_id}")
        self.terms[term_df.term_id] = term_df

