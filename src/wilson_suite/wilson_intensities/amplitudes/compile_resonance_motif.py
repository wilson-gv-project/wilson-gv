
from wilson_suite.wilson_intensities.amplitudes.term_parts import ParameterSet, ResonanceMotif, VibStatesData
from wilson_suite.wilson_intensities.amplitudes.vibene_differences import VibDiff, VibDiffCache
from .numerical_abstractions import NumericalResonanceCondition, NumericalResonanceMotif
from wilson_suite.wilson_utils.unit_convertor import convNu2Ene

def compile_resonance_motif(res_motif: ResonanceMotif,
                            param_set: ParameterSet,
                            vib_data: VibStatesData,
                            vibdiff_cache: 'VibDiffCache') -> NumericalResonanceMotif:
    """
    ResonanceMotif with ParameterSet, VibStatesData, VibDiffCache to NumericalResonanceMotif
    """
    compiled = []

    for rc in res_motif:
        vd = VibDiff.from_symbolic(rc.diff, param_set, vib_data)
        vd.cache_it(vibdiff_cache)

        energy = convNu2Ene(vd.energy_difference(au=False))
        pf = rc.pf_dict

        compiled.append(NumericalResonanceCondition(
            pf_dict=pf,
            vib_energy_diff=energy
        ))

    return NumericalResonanceMotif(res_conds=compiled)