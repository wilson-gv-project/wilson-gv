from wilson_suite.wilson_intensities.amplitudes.grid_manager_evaluator import GridRegion
from matplotlib import pyplot as plt
import matplotlib.patches as patches


def plot_regions(regions: list['GridRegion'], show: bool = False, figname = 'regions_plot.pdf'):
    
    colors = {'a+b,a': 'blue', 'b,a': 'green'}
    # region_colors = plt.cm.get_cmap('Pastel1', len(regions))
    cmap = plt.get_cmap('tab10')
    region_colors = [cmap(i) for i in range(len(regions))]
    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
    labeled_types = set()

    for i, region in enumerate(regions):

        min_a, max_a = region.domain.box.bounds['A'][0], region.domain.box.bounds['A'][1]
        min_b, max_b = region.domain.box.bounds['B'][0], region.domain.box.bounds['B'][1]
        
        # 2. Draw a background rectangle for the region
        padding = 0.5
        rect = patches.Rectangle(
            (min_a - padding, min_b - padding), 
            (max_a - min_a) + 2*padding, 
            (max_b - min_b) + 2*padding,
            linewidth=1, edgecolor=region_colors[i], facecolor=region_colors[i],
            alpha=0.15, zorder=1  # Keep it in the background
        )
        ax.add_patch(rect)
        
        # 3. Label the Region
        ax.text(min_a, max_b + padding, f"Region {i+1}", 
                fontsize=10, fontweight='bold', color=region_colors[i])

        
        for full_feat in region.domain.full_features:
            if full_feat.amplitude_coeff!=0.:
                a = full_feat.term_contributions[0].states_parameters[0]['a']
                b = full_feat.term_contributions[0].states_parameters[0]['b']
                res_type = full_feat.term_contributions[0].res_motif.to_str()

                color = colors.get(res_type, 'white')
                A = full_feat.location['A']
                B = full_feat.location['B']
                amp = full_feat.amplitude_coeff
                # ax.scatter(A, B, color=color, s=abs(amp)*1000,
                #         alpha=1.0, edgecolors='black', linewidths=1.0, zorder=5)
                
                if res_type not in labeled_types:
                    current_label = res_type
                    labeled_types.add(res_type)
                else:
                    current_label = None

                ax.scatter(A, B, color=color, s=3., label=current_label)
                # ax.annotate(f"({a},{b})", (A, B),
                #             fontsize=7, color='white',
                #             xytext=(5, 5), textcoords='offset points',
                #             bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.5))
        ax.legend(title="Resonance Type", bbox_to_anchor=(1.05, 1)) # , loc='upper left'
        # ax.set_ylim(0.)
        plt.axhline(y=0., color='k', linewidth=0.5)

    if show:
        plt.show()
    else:
        fig.savefig(figname, bbox_inches='tight', dpi=200, format='pdf')
    return fig, ax