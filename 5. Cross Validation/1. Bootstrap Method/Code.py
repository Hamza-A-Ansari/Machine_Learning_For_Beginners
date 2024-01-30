from sklearn.utils import resample

data = [1, 2, 3, 4, 5, 6]

outputBoot_resample = resample(data, replace=True, n_samples=4, random_state=1)
print('Bootstrap Sample: %s' % outputBoot_resample)

result = [x for x in data if x not in outputBoot_resample]
print('OOB Sample: %s' % result)
