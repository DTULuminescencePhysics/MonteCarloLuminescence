from __future__ import annotations
import numpy as np
from math import ceil
import matplotlib as mpl
from matplotlib import pyplot as plt
from src.classes.constants import time_to_seconds
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from src.classes.output.results_file import output_file

mpl.rcParams['font.family']='DejaVu Sans'
plt.rcParams['font.size']=18
plt.rcParams['axes.linewidth']=2

colors = ['g','b','k','c','m','y','r']
lines  = ['-','--','-.',':']

def get_timesteps_from_nan(results) -> np.ndarray:
    """Takes intermediate results data and finds when each 
    repetition ends i.e. time inputs are nan not floats"""
    times = results[:, 0, :]
    return np.sum(~np.isnan(times), axis=1)

def compute_mean_fill_from_results(results, quantiles=(0.1, 0.5, 0.9)):

    reps, _, _ = results.shape
    lengths = get_timesteps_from_nan(results)
    
    timeSteps = np.unique(np.concatenate([results[r,0,:lengths[r]]for r in range(reps) if lengths[r]>0]))
    timeSteps.sort()
    n_bins = timeSteps.size-1
    codes = np.array((0,1,2,3,4,5,6,7,8,9,10,11,12,13,14))
    counts = np.zeros((n_bins,codes.size),dtype=float)
    lumin = np.zeros((n_bins,2),dtype=float)
    q_lists = {q: [] for q in quantiles}
    fill_mean = np.zeros(n_bins)
    fill_std = np.zeros(n_bins)
    i = 0
    fill_vals = np.zeros(reps)
    for a in timeSteps[:-1]:
        j= -1
        for r in range(reps):
            L = lengths[r]
            if L <= 0:
                continue

            run_end = results[r, 0, L-1]
            if not (a < run_end):
                continue

            idx = np.searchsorted(results[r, 0, :L], a, side="right") - 1
            if idx >= 0:
                j+=1
                fill_vals[j] = results[r, 1, idx]
                counts[i, int(results[r, 3, idx])] += 1.0
               
        if j> 0:
            fill_mean[i] = np.mean(fill_vals[:j+1])
            fill_std[i] = np.std(fill_vals[:j+1], ddof=1)
            counts[i, :] /= (j+1)
            for q in quantiles:
                q_lists[q].append(np.quantile(fill_vals[:j+1], q))
        i+=1
    
    lumin[:,0] = counts[:, (np.arange(counts.shape[1]) > 1) & (np.arange(counts.shape[1]) % 2 == 0)].sum(axis=1)
    lumin[:,1] = counts[:,1]

    results = {
        "timeSteps": timeSteps,
        "meanTrapRatio": fill_mean,
        "stdTrapRatio": fill_std,
        "quantiles": {q: np.asarray(v) for q, v in q_lists.items()},
        "luminescence": lumin[:,0],
        "filling": lumin[:,1],
        "events": counts
    }

    return results


def clean_up_results(results, crystal, resultsFile: output_file, experiment_step: int = 10):
    """Build cumulative output experiments from intermediate MC results.

    For example, if ``results`` contains N repetitions and ``experiment_step`` is 10,
    this writes experiments based on the first 10, 20, 30, ... repetitions, ending
    with all N repetitions. If N is not an exact multiple of ``experiment_step``, the
    final experiment still uses all N repetitions.
    """
    reps = results.shape[0]
    if experiment_step <= 0:
        raise ValueError("experiment_step must be a positive integer")

    repetition_counts = list(range(experiment_step, reps + 1, experiment_step))
    if not repetition_counts or repetition_counts[-1] != reps:
        repetition_counts.append(reps)

    for rep_count in repetition_counts:
        cleaned_results = compute_mean_fill_from_results(results[:rep_count])
        cleaned_results["temperature"] = crystal.Tat(cleaned_results["timeSteps"])
        cleaned_results["repetitions"] = rep_count
        resultsFile.output_data_build(cleaned_results)
   

def plot_crystal(trap_coords: np.ndarray, hole_location: np.ndarray, nearest: np.ndarray, N: int, nearest_no=3):
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(trap_coords[:,0], trap_coords[:,1], trap_coords[:,2], c='blue', label='Traps')
    ax.scatter(hole_location[:,0], hole_location[:,1], hole_location[:,2], c='red', label='List2')
    
    for i in range(N):
        hole = hole_location[nearest[i,0]]
        if np.all(hole == trap_coords[i,:]):
            ax.scatter(trap_coords[i,0], trap_coords[i,1], trap_coords[i,2], c='green', label='Traps')
            continue
        xs = [trap_coords[i,0], hole[0]]
        ys = [trap_coords[i,1], hole[1]]
        zs = [trap_coords[i,2], hole[2]]
        ax.plot(xs, ys, zs, 'k--')
    
    if nearest_no > 1:
        for i in range(N):
            hole = hole_location[nearest[i,1]]
            xs = [trap_coords[i,0], hole[0]]
            ys = [trap_coords[i,1], hole[1]]
            zs = [trap_coords[i,2], hole[2]]
            ax.plot(xs, ys, zs, ls='--',color="green") 
    
    if nearest_no > 2:
        for i in range(N):
            hole = hole_location[nearest[i,2]]
            xs = [trap_coords[i,0], hole[0]]
            ys = [trap_coords[i,1], hole[1]]
            zs = [trap_coords[i,2], hole[2]]
            ax.plot(xs, ys, zs, ls='--',color="pink")

    plt.show()

def closest_divisor(N: int) -> int:
    """Finds the closest divisor to split the available MC
    runs into"""
    if N == 0: 
        return 0 
    
    if N <101:
        di = 10 
    else: 
        di = ceil(N/10)
    
    candidates = range (1, N+1) 
    divisors = [x for x in candidates if N % x == 0]

    x_best = min(divisors, key=lambda x: abs(x-di))
    return x_best

def time_sequence(input_temps, unit):
    if unit != 's':
        if unit == 'Ma':
            temp = input_temps/time_to_seconds[unit]
            last = temp[-1]
            return (((temp-last)*-1))
        else:
            return (input_temps/time_to_seconds[unit])
    else: 
        return input_temps

 



