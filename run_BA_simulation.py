import os
import time
import pandas as pd
import numpy as np
from multiprocessing import Pool
from tqdm import tqdm
from BA_simulation_module import run_single_simulation  # 외부 정의

SAVE_EVERY = 50
NUM_SIMULATIONS = 500
NUM_WORKERS = 8
N_LIST = [625, 1250, 2500]  # 자동으로 n 바꿔가며 실행 

def run_single_simulation_wrapper(args):
    i, n = args
    seed = 42 + i
    m = np.int64(n*0.0025) # 2  # 1,  3,  6
    result = run_single_simulation(n=n, m = m, S=1, seed=seed, n_estimators=200)
    result["sim_id"] = i
    result["n"] = n
    return result

def run_many_simulations_for_n(n, num_simulations, num_workers, save_every):
    output_path = f"sim_results_n{n}.csv"
    all_results = []

    if os.path.exists(output_path):
        print(f"Loading existing results from {output_path}...")
        df_existing = pd.read_csv(output_path)
        completed = set(df_existing["sim_id"])
        all_results = df_existing.to_dict("records")
    else:
        completed = set()

    tasks = [(i, n) for i in range(num_simulations) if i not in completed]

    with Pool(processes=num_workers) as pool:
        for result in tqdm(pool.imap_unordered(run_single_simulation_wrapper, tasks), total=len(tasks), desc=f"n={n}"):
            all_results.append(result)

            if (len(all_results) % save_every == 0) or (len(all_results) == num_simulations):
                df = pd.DataFrame(all_results)
                df.to_csv(output_path, index=False)
                print(f"[n={n}] Saved {len(all_results)}/{num_simulations} results to {output_path}")

    return pd.DataFrame(all_results)

if __name__ == "__main__":
    start_time = time.time()

    for n in N_LIST:
        print(f"\nStarting simulations for n = {n}")
        run_many_simulations_for_n(
            n=n,
            num_simulations=NUM_SIMULATIONS,
            num_workers=NUM_WORKERS,
            save_every=SAVE_EVERY
        )
        print(f"\nFor sample size {n} simulations complete. Total time: {(time.time() - start_time) / 3600:.2f} hours.")

    total_time = time.time() - start_time
    print(f"\nAll simulations complete. Total time: {total_time / 3600:.2f} hours.")
