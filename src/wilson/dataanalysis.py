import numpy as np
from wilson.spectrum import rec_cm2rec_s
import sys
import pandas as pd
pd.set_option('display.max_rows', sys.maxsize)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', 2000)
pd.set_option('display.width', 5000)

def get_resonances(electrical_terms_dict, mechanical_terms_dict, w_all, combos, rec_cm=True):
    """
    """
    # nfunds = len([i for i in w_all if len(i) == 1])
    w_all[('zero',)] = 0.

    dfs4terms_el = []
    dfs4terms_mech = []

    for elTerm in electrical_terms_dict:
        subscripts = electrical_terms_dict[elTerm]
        m1n1m2n2 = [i.split(',') for i in subscripts]
        letters = ['a', 'b', 'zero']
        # print(electrical_terms_dict[elTerm])
        dict_df_term = {'res': '__'.join(electrical_terms_dict[elTerm]), 'a': [], 'b': [],
                        'ω_a': [], r'ω_b': [], 'ω_2-ω_1': [], 'ω_1': [], 'ω_2': []}
        for c in combos[0]:
            dictabc = dict(zip(letters, tuple(c) + tuple(['zero'])))

            wm1 = m1n1m2n2[0][0].split('+')
            wn1 = m1n1m2n2[0][1].split('+')
            key_m = tuple(sorted([str(dictabc[i]) for i in wm1], key=int)) if 'zero' not in wm1 else tuple(wm1)
            key_n = tuple(sorted([str(dictabc[i]) for i in wn1], key=int)) if 'zero' not in wn1 else tuple(wn1)

            firstres = 1*(w_all[key_m] - w_all[key_n])

            wm1 = m1n1m2n2[1][0].split('+')
            wn1 = m1n1m2n2[1][1].split('+')
            key_m = tuple(sorted([str(dictabc[i]) for i in wm1], key=int)) if 'zero' not in wm1 else tuple(wm1)
            key_n = tuple(sorted([str(dictabc[i]) for i in wn1], key=int)) if 'zero' not in wn1 else tuple(wn1)

            secondres = -1*(w_all[key_m] -w_all[key_n])

            dict_df_term['a'].append(c[0])
            dict_df_term['b'].append(c[1])

            if rec_cm:
                dict_df_term['ω_a'].append(w_all[tuple([str(c[0])])])
                dict_df_term['ω_b'].append(w_all[tuple([str(c[1])])])
                dict_df_term['ω_2-ω_1'].append(firstres)
                dict_df_term['ω_1'].append(secondres)
                dict_df_term['ω_2'].append(firstres+secondres)
            else:
                dict_df_term['ω_a'].append(rec_cm2rec_s(w_all[tuple([str(c[0])])]))
                dict_df_term['ω_b'].append(rec_cm2rec_s(w_all[tuple([str(c[1])])]))
                dict_df_term['ω_2-ω_1'].append(rec_cm2rec_s(firstres))
                dict_df_term['ω_1'].append(rec_cm2rec_s(secondres))
                dict_df_term['ω_2'].append(rec_cm2rec_s(firstres+secondres))

        dfs4terms_el.append(dict_df_term)

    for r in dfs4terms_el:
        dd = pd.DataFrame(data=r)
        print(dd)

    for mechTerm in mechanical_terms_dict:
        subscripts, fermi = mechanical_terms_dict[mechTerm]
        m1n1m2n2 = [i.split(',') for i in subscripts]
        fermi = [i.split(',') for i in fermi]
        letters = ['a', 'b', 'c', 'zero']
        # print(mechanical_terms_dict[mechTerm])
        dict_df_term = {'res1': '__'.join(mechanical_terms_dict[mechTerm][0]), 'res2': '__'.join(mechanical_terms_dict[mechTerm][1]),
                        'a': [], 'b': [], 'c': [],
                        'ω_a': [], r'ω_b': [], 'ω_2-ω_1': [], 'ω_1': [], 'ω_2': [], 'FR1': [], 'FR2': []}
        for c in combos[1]:
            dictabc = dict(zip(letters, tuple(c) + tuple(['zero'])))

            wm1 = m1n1m2n2[0][0].split('+')
            wn1 = m1n1m2n2[0][1].split('+')
            key_m = tuple(sorted([str(dictabc[i]) for i in wm1], key=int)) if 'zero' not in wm1 else tuple(wm1)
            key_n = tuple(sorted([str(dictabc[i]) for i in wn1], key=int)) if 'zero' not in wn1 else tuple(wn1)

            firstres = 1*(w_all[key_m] -w_all[key_n])

            wm1 = m1n1m2n2[1][0].split('+')
            wn1 = m1n1m2n2[1][1].split('+')
            key_m = tuple(sorted([str(dictabc[i]) for i in wm1], key=int)) if 'zero' not in wm1 else tuple(wm1)
            key_n = tuple(sorted([str(dictabc[i]) for i in wn1], key=int)) if 'zero' not in wn1 else tuple(wn1)

            secondres = -1*(w_all[key_m] -w_all[key_n])

            wm1 = fermi[0][0].split('+')
            wn1 = fermi[0][1].split('+')
            key_m = tuple(sorted([str(dictabc[i]) for i in wm1], key=int)) if 'zero' not in wm1 else tuple(wm1)
            key_n = tuple(sorted([str(dictabc[i]) for i in wn1], key=int)) if 'zero' not in wn1 else tuple(wn1)

            thirdres = 1*(w_all[key_m] -w_all[key_n])

            wm1 = fermi[1][0].split('+')
            wn1 = fermi[1][1].split('+')
            key_m = tuple(sorted([str(dictabc[i]) for i in wm1], key=int)) if 'zero' not in wm1 else tuple(wm1)
            key_n = tuple(sorted([str(dictabc[i]) for i in wn1], key=int)) if 'zero' not in wn1 else tuple(wn1)

            fourthres = 1*(w_all[key_m] -w_all[key_n])

            dict_df_term['a'].append(c[0])
            dict_df_term['b'].append(c[1])
            dict_df_term['c'].append(c[2])

            if rec_cm:
                dict_df_term['ω_a'].append(w_all[tuple([str(c[0])])])
                dict_df_term['ω_b'].append(w_all[tuple([str(c[1])])])
                dict_df_term['ω_2-ω_1'].append(firstres)
                dict_df_term['ω_1'].append(secondres)
                dict_df_term['ω_2'].append(firstres+secondres)
                dict_df_term['FR1'].append(thirdres)
                dict_df_term['FR2'].append(fourthres)
            else:
                dict_df_term['ω_a'].append(rec_cm2rec_s(w_all[tuple([str(c[0])])]))
                dict_df_term['ω_b'].append(rec_cm2rec_s(w_all[tuple([str(c[1])])]))
                dict_df_term['ω_2-ω_1'].append(rec_cm2rec_s(firstres))
                dict_df_term['ω_1'].append(rec_cm2rec_s(secondres))
                dict_df_term['ω_2'].append(rec_cm2rec_s(firstres+secondres))
                dict_df_term['FR1'].append(rec_cm2rec_s(thirdres))
                dict_df_term['FR2'].append(rec_cm2rec_s(fourthres))
        dfs4terms_mech.append(dict_df_term)

    for r in dfs4terms_mech:
        dd = pd.DataFrame(data=r)
        print(dd)
        dd['abs(FR1)'] = abs(dd['FR1'])
        dd['abs(FR2)'] = abs(dd['FR2'])
        print('\n', dd.nsmallest(3, 'abs(FR1)'))
        print('--------------')
        print(dd.nsmallest(3, 'abs(FR2)'))

    return

def get_MechEl_contributions():
    """

    :return:
    """