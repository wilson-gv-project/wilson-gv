from dataclasses import dataclass
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from wilson_suite.wilson_intensities.amplitudes.term_parts import ParameterSet, ResonanceMotif, VibStatesData
    from wilson_suite.wilson_intensities.amplitudes.spectrum_composition import SpectralFeature
    from wilson_suite.wilson_intensities.amplitudes.vibene_differences import VibDiffCache
from wilson_suite.wilson_intensities.amplitudes.vibene_differences import VibDiff

@dataclass
class NumericalResonanceCondition:
    """
    pf_dict = {'A': meshgrid_A, 'B': meshgrid_B} - all that contribute in this resonance condition (to be summed)
    vib_energy_diff part of res condition  
        
    pfreq = sum(meshgrids[ax] * res_conds.pf_dict[ax] for ax in res_conds.pf_dict)
    
    The resonance would be at vib_energy_diff - pfreq = 0, so sum of pf_dict meshgrids = vib_energy_diff
    """
    pf_dict: dict[str, float]    # per axis scaling
    vib_energy_diff: float        # number, precomputed


@dataclass
class NumericalResonanceMotif:
    res_conds: list[NumericalResonanceCondition]


def compile_resonance_motif(res_motif: 'ResonanceMotif',
                            param_set: 'ParameterSet',
                            vib_data: 'VibStatesData',
                            vibdiff_cache: 'VibDiffCache') -> NumericalResonanceMotif:
    """
    ResonanceMotif with ParameterSet, VibStatesData, VibDiffCache to NumericalResonanceMotif
    """
    compiled = []

    for rc in res_motif:
        vd = VibDiff.from_symbolic(rc.diff, param_set, vib_data)
        vd.cache_it(vibdiff_cache)

        energy = vd.energy_difference(au=True)
        pf = rc.pf_dict

        compiled.append(NumericalResonanceCondition(
            pf_dict=pf,
            vib_energy_diff=energy
        ))

    return NumericalResonanceMotif(res_conds=compiled)


@dataclass
class CompiledTermGroup:
    resonance_motifs: list[NumericalResonanceMotif]   # one per ParameterSet


def compile_feature(feature: 'SpectralFeature',
                    vib_data: 'VibStatesData',
                    vibdiff_cache: 'VibDiffCache') -> list[CompiledTermGroup]:
    """
    
    """
    if feature.term_contributions is None:
        raise ValueError("This feature cannot be compiled without term_contributions attributed to it")
    
    compiled_groups = []

    for term_group in feature.term_contributions:
        compiled_motifs = []

        for param_set in term_group.states_parameters:
            compiled_motif = compile_resonance_motif(
                term_group.res_motif,
                param_set,
                vib_data,
                vibdiff_cache
            )
            compiled_motifs.append(compiled_motif)

        compiled_groups.append(CompiledTermGroup(compiled_motifs))

    return compiled_groups
