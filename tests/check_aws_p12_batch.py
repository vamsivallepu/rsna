"""Fixed AWS batches must reject extra work and excess spend."""
import copy
from scripts.aws_p12_batch import validate_batch

jobs = [{'recipe': r, 'seed': s, 'fold': f}
        for f, seeds in [(2, [20260918]), (4, [20260917, 20260918])]
        for s in seeds for r in ['fastmri-report', 'duration16']]
batch = dict(batch_id='remaining-confirmation', jobs=jobs,
             maximum_hours=36, compute_ceiling_usd=43.632)
assert validate_batch(batch) == 'results/final'
for change in [lambda b: b['jobs'].pop(),
               lambda b: b['jobs'].append(b['jobs'][0]),
               lambda b: b.update(maximum_hours=37),
               lambda b: b.update(compute_ceiling_usd=44)]:
    bad = copy.deepcopy(batch)
    change(bad)
    try:
        validate_batch(bad)
    except ValueError:
        pass
    else:
        raise AssertionError('Invalid batch accepted')
print('AWS fixed batch bounds passed')
