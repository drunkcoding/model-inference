import numpy as np
from tqdm import tqdm

def monte_carlo_execute(func, bounds, dtype, n=100):
    # print(bounds)
    rnd = [np.random.uniform(b_l, b_h+0.1*b_h, n).tolist() for b_l, b_h in bounds]
    rnd_choices = [
        [rnd[i][np.random.randint(0, n)] for i in range(len(bounds))]
        for _ in range(n)
    ]

    return np.array(rnd_choices), np.array([func(r) for r in rnd_choices], dtype=dtype)

def monte_carlo_bounds(func, bounds, dtype, n=100, maxiter=100, tops=10, decay=0.1):
    hist_func_out = None
    hist_func_in = None
    for _ in tqdm(range(maxiter), desc="MC Search"):
        func_in, func_out  = monte_carlo_execute(func, bounds, dtype, n)
        # print('func_in', func_in)
        # print('func_out', func_out)
        if hist_func_out is None:
            hist_func_out = func_out
            hist_func_in = func_in
        else:
            hist_func_out = np.append(hist_func_out, func_out, axis=0)
            hist_func_in = np.append(hist_func_in, func_in, axis=0)
        
        idx = np.argpartition(hist_func_out, -tops, order=[d[0] for d in dtype])[-tops:]
        # print("idx", idx)
        bounds_sample = hist_func_in[idx]
        # print("bounds_sample", bounds_sample)
        # print("func_out", hist_func_out[idx])
        
        new_bounds = list(zip(np.min(bounds_sample, axis=0), np.max(bounds_sample, axis=0)))
        # print(new_bounds, func_in)
        # assert len(new_bounds) == len(new_bounds)
        bounds = new_bounds

        hist_func_out = hist_func_out[idx]
        hist_func_in = hist_func_in[idx]

        # print('hist_func_out', hist_func_out)
        # print('hist_func_in', hist_func_in)

        if np.all(np.round(hist_func_in, 3) == np.mean(np.round(hist_func_in, 3), axis=0)):
            break

        tops = max(int(tops * (1-decay)), 1)

    return bounds
