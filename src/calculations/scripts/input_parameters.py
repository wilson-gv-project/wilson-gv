## Molecules
molecules = {'coh2': """O
C 1 B1*
H 2 B2* 1 A1*
H 2 B2* 1 A1* 3 D1

B1   =        1.215105045020483
B2   =        1.118923051621344
A1   =      122.394541535245793
D1   =      180.000000000000000
""", 'co2h2': """C
O 1 B1*
O 1 B2* 2 A1*
H 2 B3* 1 A2* 3 D1*
H 1 B4* 2 A1* 4 D2*

B1 =   1.43000000
B2 =   1.25840000
B3 =   0.96000000
B4 =   1.07000000
A1 = 120.00000000
A2 = 109.47122063
D1 = 180.00000000
D2 =   0.00000000
""", 'conh3': """O
N 1 B1*
C 1 B2* 2 A2*
H 3 B3* 1 A3* 2 D3
H 2 B4* 1 A4* 3 D3
H 2 B5* 1 A5* 3 D5

B1 =   2.27069
B2 =   1.22073
A2 =  30.02242
B3 =   1.10193
A3 = 123.65523
D3 = 180.00000
B4 =   1.01165
A4 =  91.37899
B5 =   1.00975
A5 = 149.39801
D5 =   0.00000
""", 'ch3oh': """C
O 1 B1*
H 1 B2* 2 A1*
H 2 B3* 1 A2* 3 D1*
H 1 B2* 2 A1* 4 D2*
H 1 B5* 2 A4* 4 D3*

B1 =   1.43000000
B2 =   1.07000002
B3 =   0.96000000
B5 =   1.06999998
A1 = 109.47120192
A2 = 109.50000006
A4 = 109.47120288
D1 = -60.01106737
D2 =  60.01106737
D3 = 180.00000000
""", 'coh2_h2o': """O
C 1 B1*
H 2 B2* 1 A1*
H 2 B2* 1 A1* 3 D1*
O 2 B4* 1 A3* 3 D2*
H 5 B5* 2 A4* 1 D3*
H 5 B5* 6 A5* 2 D4*

B1 =    1.25840000
B2 =    1.07000000
B4 =    1.57027396
B5 =    0.96000000
A1 =  120.00000000
A3 =   93.50834554
A4 =   91.64193728
A5 =  109.47122063
D1 = -180.00000000
D2 =  -95.17367001
D3 =  -49.23014824
D4 =  -83.97041690
""", 'ch3coh': """C
C 1 B1*
O 1 B2* 2 A1
H 2 B3* 1 A2 3 D1
H 2 B3* 1 A2 3 D2
H 2 B3* 1 A2 3 D3
H 1 B3* 2 A1 4 D4

B1   =        1.519585561487551
B2   =        1.215668722310588
A1   =      119.999999999999986
B3   =        1.106170788021851
A2   =      109.471220630000019
D1   =      -30.000000000000092
D2   =       90.000000000000000
D3   =     -149.999999999999886
D4   =      149.999999999999773
"""}

## Optimization
settingsOpt = {'level of theory': tuple(['CCSD(T)', 'cc-pVQZ']),
                'geoconv':11, 'ccconv':11, 'scfconv':11, 'lineqconv':11, 'geocycles': 70,
                'scfcycles': 250, 'cccycles': 250, 'lineqcycles': 150}

configHPCopt = {'machine':'fram', 'minutes':'30', 'hours':'27', 'nodes':1, 'dir3':False,
                'c4path': '/cluster/projects/nn14654k/vle014/cfour_serial/bin'}

## Anharmonic parallel
settingsCalc = {'level of theory': tuple(['HF', 'cc-pVQZ']),
                'geoconv':12, 'ccconv':12, 'scfconv':12, 'lineqconv':12, 'geocycles': 50,
                'scfcycles': 390, 'cccycles': 350, 'lineqcycles': 300,
                'job':'ANHARM=VPT2\nANH_ALGORITHM=PARALLEL\nVIBRATION=ANALYTIC\nFD_PROJECT=ON\nPRINT=1',
                'jobtype':'anharm'}

# main job
configHPC = {'machine':'fram', 'minutes':'40', 'hours':'08', 'nodes':1, 'dir3':True,
             'c4path': '/cluster/projects/nn14654k/vle014/cfour_serial/bin'}
# generated displaced zmat0*
configHPCdispl = {'machine':'fram', 'minutes':'30', 'hours':'18', 'nodes':1, 'dir3':True,
                  'c4path': '/cluster/projects/nn14654k/vle014/cfour_serial/bin'}

# for fja files processing
config_fja = {'machine':'fram', 'minutes':'05', 'hours':'00', 'nodes':1, 'c4path': '/cluster/projects/nn14654k/vle014/cfour_serial/bin'}

## polarizability
settingsCalcPolar = {'level of theory': tuple(['CCSD(T)', 'cc-pVQZ']),
                     'geoconv':11, 'ccconv':11, 'scfconv':11, 'lineqconv':11, 'geocycles': 50,
                     'scfcycles': 400, 'cccycles': 350, 'lineqcycles': 300,
                     'job':'PROPS=SECOND_ORDER\nPRINT=1',
                     'jobtype':'polar'}

configHPCpolar = {'machine':'fram', 'minutes':20, 'hours':'18', 'nodes':1, 'dir3':True, 'c4path': '/cluster/projects/nn14654k/vle014/cfour_serial/bin'}
