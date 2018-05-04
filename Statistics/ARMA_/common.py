# -*- coding:utf8 -*-

import numpy as np

def r_squared(real, resid):
	mean = np.tile(np.average(real), (1, real.shape[0]))
	SS_tot = np.sum( (mean - real) ** 2 )
	SS_reg = np.sum( (mean - resid) ** 2 )
	print SS_reg * 1.0 / SS_tot
