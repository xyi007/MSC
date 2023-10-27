import eval_overall

import logging

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Evaluate model ensemble
paths = ['/home/xy/project/MSC/checkpoint/f30k/sgr/results_f30k_precomp.npy',
         '/home/xy/project/MSC/checkpoint/f30k/saf/results_f30k_precomp.npy']

eval_overall.eval_ensemble(results_paths=paths, fold5=False)
#eval_overall.eval_ensemble(results_paths=paths, fold5=True)