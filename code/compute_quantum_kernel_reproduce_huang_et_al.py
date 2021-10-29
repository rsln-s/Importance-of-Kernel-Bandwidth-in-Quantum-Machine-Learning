"""
Computes quantum kernel and dumps everything on disk

"""

import argparse
import sys
import time
import pickle
from pathlib import Path
from sklearn.svm import SVC
from utils import (
    get_dataset, 
    get_quantum_kernel, 
    HamiltonianEvolutionFeatureMap_reproduce_Google, 
)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--outpath", type = Path,
        required = True,
        help = "folder to dump the result")
    parser.add_argument(
        "--dataset-dim", type = int,
        required = True,
        help = "dimensionality (number of qubits)")
    parser.add_argument(
        "--n-trotter", type = int,
        required = True,
        help = "number of trotter steps for the feature maps")
    parser.add_argument(
        "--init-state-seed", type = int,
        required = True,
        help = "the seed used for the random initial state")
    parser.add_argument(
        "--evo-time-factor", type = float,
        required = True,
        help = "total system evolution time is proportional to system size and given by (dataset_dim * evo_time_factor)")
    parser.add_argument(
        "--init-state", type = str,
        required = True,
        help = "initial state for the Hamiltonian evolution feature map")
    args = parser.parse_args()
    outpath = Path(args.outpath, f"reproduce_huang_et_al_dim_{args.dataset_dim}_ntrot_{args.n_trotter}_evo_t_{args.evo_time_factor}_init_{args.init_state}_s_{args.init_state_seed}.p")
    if outpath.exists():
        print(f"Found already computed at {outpath}, exiting")
        sys.exit()
    else:
        print(f"Using outpath: {outpath}")
    t0 = time.time()

    if args.dataset_dim > 30:
        raise ValueError(f"Dataset dimension {args.dataset_dim} too large; can support no more than 30 qubits")

    evo_time = args.dataset_dim * args.evo_time_factor
    
    x_train, x_test, y_train, y_test = get_dataset('fashion-mnist', args.dataset_dim, 800, 200)
    FeatureMap, scaling_factor = HamiltonianEvolutionFeatureMap_reproduce_Google(args.dataset_dim, args.n_trotter, evo_time, args.init_state, args.init_state_seed)
    x_train *= scaling_factor 
    x_test *= scaling_factor

    qkern = get_quantum_kernel(FeatureMap, batch_size=50)
    qkern_matrix_train = qkern.evaluate(x_vec=x_train)
    t1 = time.time()
    K_train_time = t1-t0
    print(f"Done computing K_train in {K_train_time}")

    qkern_matrix_test = qkern.evaluate(x_vec=x_test, y_vec=x_train)
    t2 = time.time()
    K_test_time = t2-t1
    print(f"Done computing K_test in {K_test_time}")

    qsvc = SVC(kernel='precomputed')
    qsvc.fit(qkern_matrix_train, y_train)
    score = qsvc.score(qkern_matrix_test, y_test)
    print(f"Score: {score}\n Error: {1-score}\n for {args}")

    res = {
            'qkern_matrix_train' : qkern_matrix_train,
            'qkern_matrix_test' : qkern_matrix_test,
            'score' : score,
            'args' : args,
            'K_train_time' : K_train_time,
            'K_test_time' : K_test_time,
    }
    pickle.dump(res, open(outpath, 'wb'))
    print(f"Dumped the result to {outpath}")
