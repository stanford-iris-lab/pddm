import logging

logging.disable(logging.CRITICAL)
import numpy as np
import multiprocessing as mp
import time as timer

# my imports
from pddm.samplers import base_sampler


def sample_paths_parallel(
    N,
    actions_list,
    actions_taken_so_far,
    starting_fullenvstate,
    env,
    max_process_time=300,
    max_timeouts=4,
    suppress_print=False,
):

    num_cpu = mp.cpu_count()
    paths_per_cpu = int(
        np.floor(N / num_cpu)
    )  # round down, so last CPU isn't trying to index things that don't exist

    args_list = []
    for i in range(num_cpu):
        which_cpu = i
        args_list_cpu = [
            paths_per_cpu,
            actions_list,
            actions_taken_so_far,
            starting_fullenvstate,
            which_cpu,
            env,
        ]
        args_list.append(args_list_cpu)

    # Do multiprocessing
    if not suppress_print:
        start_time = timer.time()
        print("####### Gathering Samples #######")

    results = _try_multiprocess(args_list, num_cpu, max_process_time, max_timeouts)

    # result is paths type, and results is list of paths
    paths = []
    for result in results:
        for path in result:
            paths.append(path)

    if not suppress_print:
        print(
            "======= Samples Gathered  ======= | >>>> Time taken = %f "
            % (timer.time() - start_time)
        )

    return paths


def _try_multiprocess(args_list, num_cpu, max_process_time, max_timeouts):

    # Base case
    if max_timeouts == 0:
        return None

    pool = mp.Pool(processes=num_cpu, maxtasksperchild=1)
    parallel_runs = [
        pool.apply_async(base_sampler.do_rollout_star, args=(args_list[i],))
        for i in range(num_cpu)
    ]

    try:
        results = [p.get(timeout=max_process_time) for p in parallel_runs]
    except Exception as e:
        print(str(e))
        print("Timeout Error raised... Trying again")
        pool.close()
        pool.terminate()
        pool.join()
        return _try_multiprocess(args_list, num_cpu, max_process_time, max_timeouts - 1)

    pool.close()
    pool.terminate()
    pool.join()
    return results
