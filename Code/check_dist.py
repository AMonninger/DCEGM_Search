from HARK.distribution import MeanOneLogNormal
d = MeanOneLogNormal(sigma=1.0)
disc = d.discretize(7)
print(type(disc))
print(dir(disc))
print('atoms:', disc.atoms)
print('pmv:', disc.pmv)
