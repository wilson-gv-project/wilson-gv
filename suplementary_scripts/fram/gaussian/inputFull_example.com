%chk=inputFull_3q
%mem=745MB
#P B3LYP/cc-pVQZ Opt=VeryTight
        Int=UltraFine SCF=VeryTight

 Title

0 1
C
O 1 B1
H 1 B2 2 A1
H 2 B3 1 A2 3 D1
H 1 B2 2 A1 4 D2
H 1 B5 2 A4 4 D3

B1   =        1.415645818227121
B2   =        1.090720111867709
A1   =      112.112045618910358
B3   =        0.955921012421149
A2   =      108.100097058206302
D1   =      -61.453767004057994
D2   =       61.453767004057994
B5   =        1.085179192784929
A4   =      106.854430345921742
D3   =      180.000000000000000

--Link1--
%chk=inputFull_3q
%mem=745MB
#P B3LYP/cc-pVQZ Freq=(Anharmonic,raman,hpmodes,ReadAnharm)
        Int=UltraFine SCF=VeryTight
        iop(7/33=1) IOp(10/96=2)
        Guess=Read Geom=AllCheck

Spectro=MaxQuanta=3

