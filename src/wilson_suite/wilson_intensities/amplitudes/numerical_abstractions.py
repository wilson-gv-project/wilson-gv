from dataclasses import dataclass

from wilson_suite.wilson_intensities.amplitudes.term_parts import ParameterSet, ResonanceMotif, VibStatesData
from wilson_suite.wilson_intensities.amplitudes.vibene_differences import VibDiff, VibDiffCache
from wilson_suite.wilson_utils.unit_convertor import convNu2Ene


@dataclass
class NumericalResonanceCondition:
    pf_dict: dict[str, float]    # per axis scaling
    vib_energy_diff: float        # number, precomputed


@dataclass
class NumericalResonanceMotif:
    res_conds: list[NumericalResonanceCondition]


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