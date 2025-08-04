import plotly.graph_objects as go
import numpy as np
from .base_template import BaseRenderer

class PlotlyRenderer(BaseRenderer):
    def create_figure(self):
        return go.Figure()

    def assign_axes(self, fig):
        return fig  # plotly operates on the figure directly

    def plot_data(self, fig):
        x, y, z = self.spectrum_data['x'], self.spectrum_data['y'], self.spectrum_data.get('z', None)
        if self.projection == "2d":
            if self.scatter or z is None:
                fig.add_trace(go.Scatter(x=x, y=y, mode='markers',
                                         marker=dict(color=z, colorbar=dict(title="Z"))))
            else:
                x_unique = np.unique(x)
                y_unique = np.unique(y)
                Z = z.reshape(len(y_unique), len(x_unique))
                fig.add_trace(go.Contour(z=Z, x=x_unique, y=y_unique, colorscale='Viridis'))
        elif self.projection == "1d":
            fig.add_trace(go.Scatter(x=x, y=y, mode='lines'))

    def set_dynamic_range(self, fig):
        if self.dynamic_range:
            fig.update_traces(zmin=self.dynamic_range[0], zmax=self.dynamic_range[1], selector=dict(type='contour'))

    def style_axes(self, fig):
        fig.update_layout(
            title="Spectrum",
            xaxis_title="X",
            yaxis_title="Y",
        )

    def finalize_figure(self, fig):
        fig.update_layout(margin=dict(l=0, r=0, t=40, b=0))
