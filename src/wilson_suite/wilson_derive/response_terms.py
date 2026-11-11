import copy
import json
from dataclasses import dataclass
from fractions import Fraction

from wilson_suite.wilson_derive.abstractions import PolProp, VibDiffTerm, ResonanceCondition, QOperator, \
    HarmOscStateSymbolic, TransitionIntegral

class VibPerturbedTerm:
    """
    Class to represent post-Hermite treatment term: Instances of this class are the main end result of a typical
    wilson-derive run.
    """

    def __init__(self, coeff: Fraction, props: list[PolProp], freqterms: list[VibDiffTerm], res: list[ResonanceCondition],
                 order_el: int=0, order_mech: int=0):
        """
        A term consists of the following attributes:

        coeff: Coefficient of term (Fraction)

        props: Molecular properties: At the stage of derivation where the present class is relevant, all of these
        would involve at least one order of geometrical differentiation (list of PolProp instances)

        freqterms: Prefactors of inverse differences between vibrational energy levels as identified
        by the derivation (list of VibDiffTerm instances)

        res: Resonance conditions as identified by the derivation (list of ResonanceCondition instances)

        order_el, order_mech: Integers: Order of electrical resp. mechanical anharmonicity: Default 0 for both
        """

        if not (isinstance(coeff, Fraction) ):
            raise TypeError('VibPerturbedTerm coeff must be a Fraction instance')
        self.coeff = coeff

        if not (all([isinstance(i, PolProp) for i in props])):
            raise TypeError('All transition integral arguments in VibPerturbedTerm must be transitionIntegral instances')
        self.props = props

        if not (all([isinstance(i, VibDiffTerm) for i in freqterms])):
            raise TypeError('All frequency arguments in VibPerturbedTerm must be vibDiffTerm instances')
        self.freqterms = freqterms

        if not (all([isinstance(i, ResonanceCondition) for i in res])):
            raise TypeError('All resonance condition arguments in VibPerturbedTerm must be resonanceCondition instances')
        self.res = res

        # Boolean: Has this term been sorted according to the below sort method?
        self.was_sorted = False

        self.order_el = order_el

        self.order_mech = order_mech

        nonsumm_ind, summ_ind = self.tellNonSummSummIndices()

        self.summ_ind = summ_ind
        self.nonsumm_ind = nonsumm_ind

        # Hash (currently indeterminate)
        self.hsh = None

    def __repr__(self):
        return f"VibPerturbedTerm(coeff = {self.coeff}, props = {self.props}, freqterms = {self.freqterms}, res = {self.res})"

    def tellNonSummSummIndices(self) -> tuple[list[str], list[str]]:
        """
        Tell which normal modeindices in this term are "non-summation" (i.e. involved in a resonance condition) and
        which are "summation" (i.e. not involved in a resonance condition)

        Returns a 2-tuple: Each entry is a list of characters referring to resp. non-summation and summation indices
        """

        summation_indices = []
        non_summation_indices = []

        # Walk through resonance conditions: Every index encountered here is a non-summation index
        for i in self.res:

            for j in i.diff.sl.q:
                if not j in non_summation_indices:
                    non_summation_indices.append(j)

            for j in i.diff.sr.q:
                if not j in non_summation_indices:
                    non_summation_indices.append(j)

        candidate_summation_indices = []

        # Walk through frequency terms: Every index here but not non-summmation is a summation inde
        for i in self.freqterms:

            for j in i.sl.q:
                if not j in candidate_summation_indices:
                    candidate_summation_indices.append(j)

            for j in i.sr.q:
                if not j in candidate_summation_indices:
                    candidate_summation_indices.append(j)

        for i in candidate_summation_indices:

            if not i in non_summation_indices:
                summation_indices.append(i)


        return non_summation_indices, summation_indices


    def nmRenameAndInternalResort(self, mask: dict):
        """
        Take a mask (dictionary of single-character key: value pairs) in an ordering to be replaced by the canonical normal mode
        index list: Example: mask is {'b': 'a', 'a': 'b', 'c': 'c'} -> Replace every reference to 'b' in self with 'a',
        replace every (original) 'a' with 'b' and leave 'c' unchanged
        """

        # In properties
        for i in range(len(self.props)):
            for j in range(len(self.props[i].inds)):
                self.props[i].inds[j] = mask[self.props[i].inds[j]]
            self.props[i].inds = sorted(self.props[i].inds)

        # In (non-resonance condition) frequency terms
        for i in range(len(self.freqterms)):
            for j in range(len(self.freqterms[i].sl.q)):
                self.freqterms[i].sl.q[j] = mask[self.freqterms[i].sl.q[j]]
            self.freqterms[i].sl.q = sorted(self.freqterms[i].sl.q)
            for j in range(len(self.freqterms[i].sr.q)):
                self.freqterms[i].sr.q[j] = mask[self.freqterms[i].sr.q[j]]
            self.freqterms[i].sr.q = sorted(self.freqterms[i].sr.q)

        # In resonance conditions
        for i in range(len(self.res)):
            for j in range(len(self.res[i].diff.sl.q)):
                self.res[i].diff.sl.q[j] = mask[self.res[i].diff.sl.q[j]]
            self.res[i].diff.sl.q = sorted(self.res[i].diff.sl.q)
            for j in range(len(self.res[i].diff.sr.q)):
                self.res[i].diff.sr.q[j] = mask[self.res[i].diff.sr.q[j]]
            self.res[i].diff.sr.q = sorted(self.res[i].diff.sr.q)

        for i in range(len(self.nonsumm_ind)):
            self.nonsumm_ind[i] = mask[self.nonsumm_ind[i]]

        for i in range(len(self.summ_ind)):
            self.summ_ind[i] = mask[self.summ_ind[i]]

    def sort(self, nm_inds):
        """
        Sort and possibly rename indices in term:
        - Sort resonance conditions in increasing order of number of perturbing frequencies
        - Rename normal mode indices according to encountered state labels in resonance conditions (currently leaving
        further indices in arbitrary order)
        - Sort operator references in properties in increasing numerolexical order
        - Sort properties among each order in increasing order of geometric differentiation
        - Then sort properties at same order of differentiation according to operator (numerolexical) ordering
        - Sort frequency difference terms internally to have greatest number of quanta in left-hand state
        - Sort terms tied wrt. previous sorting according to lexical order of state tuples
        - Sort freq diff terms wrt. each other according to lexical ordering of right-hand state indices
          (sort tied terms by left-hand state lex. ordering)

        nm_inds: List of canonically ordered normal mode indices (would normally be ['a', 'b', ...])
        """

        # Sort resonance conditions in increasing order of number of perturbing frequencies
        self.res = sorted(self.res, key=lambda j:len(j.pf))

        # Make normal mode index sorting mask:
        # First gather according to encountered state labels in resonance conditions
        # Then get remaining indices as encountered in derivatives (FIXME: Likely degree of freedom)

        nm_labels = []

        for i in self.res:

            for j in i.diff.sl.q:
                if not j in nm_labels:
                    nm_labels.append(j)

            for j in i.diff.sr.q:
                if not j in nm_labels:
                    nm_labels.append(j)

        for i in self.props:
            for j in i.inds:
                if not j in nm_labels:
                    nm_labels.append(j)

        # If the label progression doesn't correspond to the canonical progression, then rename to make canonical
        # and re-sort (FIXME: Could be remaining sorting "slack" after this)
        if not (nm_labels == nm_inds[:len(nm_labels)]):
            nm_mask = {nm_labels[i]: nm_inds[i] for i in range(len(nm_labels))}
            self.nmRenameAndInternalResort(nm_mask)

        # Internal sorting: Operators in each property in increasing order
        for i in self.props:
            i.ops = sorted(i.ops, key=lambda j: j.o)

        # Inter-term sorting

        # First sort properties according to increasing order of differentiation
        self.props = sorted(self.props, key=lambda j: j.dord)

        # Then sort properties at same order of differentiation according to operators
        # NOTE: I believe this should take care of both number and lexical ordering

        m = 0
        dord_starts = [0]

        for i in range(len(self.props)):
            if self.props[i].dord > self.props[dord_starts[m]].dord:
                dord_starts.append(i)
                m += 1

        dord_starts.append(len(self.props) - 1)

        for i in range(len(dord_starts)):
            # No need to sort if at the last entry
            if not dord_starts[i] == len(self.props) - 1:
                # No need to sort a length 1 category
                if not dord_starts[i + 1] == dord_starts[i] + 1:
                    self.props[dord_starts[i]:dord_starts[i+1]] = sorted(self.props[dord_starts[i]:dord_starts[i+1]],
                                                                         key=lambda j: [k.o for k in j.ops])

        # Sort all freq term bras and kets

        # Internal sorting (assumes all individual nm tuples already sorted)

        # Make all freq diff terms have the greatest number of quanta in bra - flip and change sign if necessary
        for i in range(len(self.freqterms)):
            if len(self.freqterms[i].sl.q) < len(self.freqterms[i].sr.q):
                self.coeff *= Fraction(-1)
                tmp = copy.deepcopy(self.freqterms[i].sl)
                self.freqterms[i].sl = copy.deepcopy(self.freqterms[i].sr)
                self.freqterms[i].sr = tmp

            # Sort tied terms according to nm indices
            elif len(self.freqterms[i].sl.q) == len(self.freqterms[i].sr.q):
                if not (sorted([self.freqterms[i].sl.q, self.freqterms[i].sr.q]) ==
                               [self.freqterms[i].sl.q, self.freqterms[i].sr.q]):
                    self.coeff *= Fraction(-1)
                    tmp = copy.deepcopy(self.freqterms[i].sl)
                    self.freqterms[i].sl = copy.deepcopy(self.freqterms[i].sr)
                    self.freqterms[i].sr = tmp

        # Inter-term sorting

        # Sort freq diff terms first according to right-hand indices
        self.freqterms = sorted(self.freqterms, key=lambda j:j.sr.q)

        # Then sort tied freq diff terms according to left-hand indices
        ketq_starts = [0]

        m = 0

        for i in range(len(self.freqterms)):
            if not (self.freqterms[i].sr.q == self.freqterms[ketq_starts[m]].sr.q):
                ketq_starts.append(i)
                m += 1

        ketq_starts.append(len(self.freqterms))

        for i in range(len(ketq_starts)):
            if not ketq_starts[i] == len(self.freqterms):
                self.freqterms[ketq_starts[i]:ketq_starts[i + 1]] = \
                    sorted(self.freqterms[ketq_starts[i]:ketq_starts[i + 1]], key=lambda j: j.sl.q)

        # Mark term as having been sorted
        # WARNING: This flag doesn't update if any subsequent changes are made
        self.was_sorted = True

    def h(self, also_sort: bool=False, nm_inds: list=None) -> int:
        """
        Hashing function

        also_sort: Also sort term before hash is calculated?
        nm_inds: List of normal mode indices if sorting
        """

        if not(self.was_sorted) and not(also_sort):
            raise AssertionError('Term for which hash was requested has not been sorted')

        if also_sort:
            self.sort(nm_inds)

        # Getting hashes of constituent parts
        props_h = tuple([hash(i) for i in self.props])
        ft_h = tuple([i.h() for i in self.freqterms])
        res_h = tuple([i.h() for i in self.res])

        # Combine constituent hashes for collective hash for this term
        self.hsh = hash((props_h, ft_h, res_h))

        return self.hsh

    def full_enhancement_possible(self, mcp: list[list]|tuple[list], mcm: list[list]|tuple[list]) -> bool:
        """
        Determine: Given the setup/requested frequency ranges, is it possible for this term to become resonant within these
        ranges with respect to all of its resonance conditions?

        TODO: Currently only for sole harmonic oscillator state, may later need extension for variationally resolved states

        mcp: List or tuple: Each entry a list: Each such list a collection of interaction indices
        whose frequency sum is imposed as positive (ordinarily from analyzing a specific interaction sequence of pulses
        in an experiment) by a margin deemed "significant" according to a previously decided threshold

        mcm: As mcp but referring to sums imposed as negative

        Returns: bool: True if full enhancement is possible (or can't be ruled out), False if full enhancement
        definitely not possible
        """

        if not self.was_sorted:
            raise ValueError('Term must be sorted before full enhancement possibility can be evaluated')

        # There is only something to do here if there are actually conditions
        if mcp is not None or mcm is not None:

            prev_res = None
            for i in self.res:

                # Try with a preceding resonance condition as given if one is available
                if prev_res is not None:

                    if not (i.could_be_resonant_by_conditions(mcp, mcm, given_prev_res=prev_res)):
                        return False

                # Also try with no preceding resonance condition (regardless of whether one is available or not)
                if not (i.could_be_resonant_by_conditions(mcp, mcm, given_prev_res=None)):
                    return False

                prev_res = copy.deepcopy(i)

            return True

        # If no information given, return True to keep term
        return True

    def present(self):
        """
        Formatted printing of own attributes
        """
        print(' >> VibPerturbedTerm presents:')

        print('Coefficient:', self.coeff)
        print('Properties:')
        for i in self.props:
            i.present()
        print('Freqterms:')
        for i in self.freqterms:
            i.present()
        print('Resonances:')
        for i in self.res:
            i.present()

        print('Electrical anharm. order:', self.order_el)
        print('Mechanical anharm. order:', self.order_mech)
        print('Indices part of resonance conditions:', self.nonsumm_ind)
        print('Further (summation) indices:', self.summ_ind)

        print('\nHas attributes: coeff, freqterms, res, was_sorted, props, hsh'
              '\nHas methods: nmRenameAndResort, sort, h, full_enhancement_possible, present'
              '\nPresenting also elements of: self.props, self.freqterms, self.res')

    def present_better(self):
        """
        Formatted printing of own attributes
        """
        print('\n >> VibPerturbedTerm presents:')

        print('Coefficient:', self.coeff)
        print('Properties:')
        for i in self.props:
            print('    ', i)
        print('Freqterms:')
        for i in self.freqterms:
            print('    ', i)
        print('Resonances:')
        for i in self.res:
            print('    ', i)

        print('\nHas attributes: coeff, freqterms, res, was_sorted, props, hsh'
              '\nHas methods: nmRenameAndResort, sort, h, full_enhancement_possible, present'
              '\nPresenting also elements of: self.props, self.freqterms, self.res')

    def to_latex(self, part=None) -> str:
        """
        Make LaTeX representation of own attributes
        """
        res_conditions_denom = ''.join([rc.to_latex() for rc in self.res])
        if res_conditions_denom == '':
            res_conditions_str = ''
        else:
            res_conditions_str = rf'\frac{{1}}{{{res_conditions_denom}}}'

        coefficients_str = rf'\frac{{{self.coeff.numerator}}}{{{self.coeff.denominator}}}'
        properties_str = ''.join([p.to_latex() for p in self.props])

        freqterms_denom = ''.join([rf'\omega_{{{vd.to_latex()}}}' for vd in self.freqterms])
        if freqterms_denom == '':
            freqterms_str = ''
        else:
            freqterms_str = rf'\frac{{{1}}}{{{freqterms_denom}}}'

        if part is not None:
            if part=='res':
                return res_conditions_str
            elif part=='coeff':
                return coefficients_str
            elif part=='props':
                return properties_str
            elif part=='freqterms':
                return freqterms_str
        else:
            return coefficients_str + freqterms_str + properties_str + res_conditions_str

    def to_dict(self) -> dict:
        """Convert VibPerturbedTerm to a dictionary for JSON serialization"""
        return {
            "coeff": [self.coeff.numerator, self.coeff.denominator],
            "order_el": self.order_el,
            "order_mech": self.order_mech,
            "props": [
                {
                    "ops": [{"o": op.o, "op_type": op.op_type, "ax": op.ax} for op in p.ops],
                    "dord": p.dord,
                    "inds": p.inds
                } for p in self.props
            ],
            "freqterms": [
                {
                    "sl": {"q": ft.sl.q},
                    "sr": {"q": ft.sr.q},
                    "is_pert_wf_diff": ft.is_pert_wf_diff
                } for ft in self.freqterms
            ],
            "res": [
                {
                    "diff": {
                        "sl": {"q": r.diff.sl.q},
                        "sr": {"q": r.diff.sr.q}
                    },
                    "pf": r.pf,
                    "id": r.id
                } for r in self.res
            ],
            "was_sorted": self.was_sorted
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'VibPerturbedTerm':
        """Create VibPerturbedTerm from a dictionary"""
        # Reconstruct coefficient
        coeff = Fraction(data["coeff"][0], data["coeff"][1])

        order_el = data["order_el"]
        order_mech = data["order_mech"]

        # Reconstruct properties
        props = []
        for p in data["props"]:
            ops = [QOperator(op["o"], op["op_type"], op["ax"]) for op in p["ops"]]
            prop = PolProp(ops, p["dord"])
            prop.inds = p["inds"]
            props.append(prop)

        # Reconstruct frequency terms
        freqterms = []
        for ft in data["freqterms"]:
            sl = HarmOscStateSymbolic(ft["sl"]["q"])
            sr = HarmOscStateSymbolic(ft["sr"]["q"])
            freqterms.append(VibDiffTerm(sl, sr, ft["is_pert_wf_diff"]))

        # Reconstruct resonance conditions
        res = []
        for r in data["res"]:
            sl = HarmOscStateSymbolic(r["diff"]["sl"]["q"])
            sr = HarmOscStateSymbolic(r["diff"]["sr"]["q"])
            diff = VibDiffTerm(sl, sr)
            res.append(ResonanceCondition(diff, r["pf"], r["id"]))

        # Create instance
        instance = cls(coeff, props, freqterms, res, order_el, order_mech)
        instance.was_sorted = data["was_sorted"]
        return instance

    def to_json(self, filepath: str = None) -> str:
        """Convert to JSON string or save to file"""
        json_str = json.dumps(self.to_dict(), indent=2)
        if filepath:
            with open(filepath, 'w') as f:
                f.write(json_str)
        return json_str

    @classmethod
    def from_json(cls, json_str: str = None, filepath: str = None) -> 'VibPerturbedTerm':
        """Create instance from JSON string or file"""
        if filepath:
            with open(filepath) as f:
                data = json.load(f)
        else:
            data = json.loads(json_str)
        return cls.from_dict(data)

    @classmethod
    def save_many_to_json(cls, terms: list['VibPerturbedTerm'], filepath: str):
        """Save multiple terms to a single JSON file"""
        terms_data = [term.to_dict() for term in terms]
        with open(filepath, 'w') as f:
            json.dump(terms_data, f, indent=2)

    @classmethod
    def load_many_from_json(cls, filepath: str) -> list['VibPerturbedTerm']:
        """Load multiple terms from a JSON file"""
        with open(filepath) as f:
            terms_data = json.load(f)
        return [cls.from_dict(term_data) for term_data in terms_data]


class VibContribTerm:
    """
    Class to represent a vibrational term in SOS expression up to but not including Hermite treatment
    """

    def __init__(self, coeff: Fraction, ints: list[TransitionIntegral], res: list[ResonanceCondition], comp_pattern: tuple[int],
                 order_el: int = 0, order_mech: int = 0):
        """
        A term consists of the following attributes:

        coeff: Coefficient of term (Fraction)

        ints: Transition integrals involving a bra state, one or more operators, and a ket state

        res: Resonance conditions as identified by the derivation (list of ResonanceCondition instances)

        comp_pattern: Compounding pattern as a tuple of integers
        """

        if not (isinstance(coeff, Fraction) ):
            raise TypeError('vibContribTerm coeff must be a Fraction instance')
        self.coeff = coeff

        if not (all([isinstance(i, TransitionIntegral) for i in ints])):
            raise TypeError('All transition integral arguments in VibContribTerm must be TransitionIntegral instances')
        self.ints = ints

        if not (all([isinstance(i, ResonanceCondition) for i in res])):
            raise TypeError('All resonance condition arguments in VibContribTerm must be ResonanceCondition instances')
        self.res = res

        if not (isinstance(comp_pattern, tuple)):
            raise TypeError('The compounding pattern argument in VibContribTerm must be a tuple of integers')
        if not (all([isinstance(i, int) for i in comp_pattern])):
            raise TypeError('The compounding pattern argument in VibContribTerm must be a tuple of integers')
        self.comp_pattern = comp_pattern

        self.order_el = order_el

        self.order_mech = order_mech

        # Initialize frequency difference terms in anticipation of possible addition
        # from mechanical anharmonicity handling
        self.freqdiff = []

    @property
    def order(self) -> int:
        # Always true for well-formed compounding patterns
        # Could alternatively get as total number of operators in transition integrals
        return sum(self.comp_pattern) - 1

    def addFreqTerm(self, new_term: VibDiffTerm):
        """
        Add (inverted) frequency difference term from mechanical anharmonicity handling
        """

        if not (isinstance(new_term, VibDiffTerm)):
            raise TypeError('VibContribTerm frequency difference terms must be VibDiffTerm instances')

        self.freqdiff.append(new_term)


    def present(self):
        """
        Formatted printing of own attributes
        """

        print('Coefficient:', self.coeff)

        for i in self.ints:
            i.present()

        for i in self.res:
            i.present()

        for i in self.freqdiff:
            i.present()

        print('')

