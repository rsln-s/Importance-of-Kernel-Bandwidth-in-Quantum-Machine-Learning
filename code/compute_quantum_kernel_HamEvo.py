"""
Computes quantum kernel and dumps everything on disk

"""

import argparse
import sys
import numpy as np
import time
import pickle
from pathlib import Path
from sklearn.svm import SVC
from utils import (
    get_dataset, 
    get_quantum_kernel, 
    HamiltonianEvolutionFeatureMap, 
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
        "--evo-time", type = float,
        required = True,
        help = "total system evolution time")
    parser.add_argument(
        "--init-state", type = str,
        required = True,
        help = "initial state for the Hamiltonian evolution feature map")
    parser.add_argument(
        "--init-state-seed", type = int,
        required = True,
        help = "the seed used for the random initial state")
    parser.add_argument(
        "--decimals", type = int,
        required = False,
        default=None,
        help = "number of decimal points to keep (passed directly to np.round)")
    parser.add_argument(
        "--dataset", type = str,
        required = True,
        choices=['fashion-mnist','kmnist','plasticc'],
        help = "dataset to use")
    parser.add_argument(
        "--simulation-method", type = str,
        required = False,
        default="statevector",
        help = "simulation method to use (passed directly to qiskit)")
    parser.add_argument(
        "--shots", type = int,
        required = False,
        default=1,
        help = "number of shots to use (passed directly to qiskit)")
    args = parser.parse_args()
    outpath = Path(args.outpath, f"dim_{args.dataset_dim}_ntrot_{args.n_trotter}_evo_t_{args.evo_time}_init_{args.init_state}_s_{args.init_state_seed}_dec_{args.decimals}_{args.dataset}_{args.simulation_method}.p")
    if outpath.exists():
        print(f"Found already computed at {outpath}, exiting")
        sys.exit()
    else:
        print(f"Using outpath: {outpath}")
    t0 = time.time()

    if args.dataset_dim > 30:
        raise ValueError(f"Dataset dimension {args.dataset_dim} too large; can support no more than 30 qubits")

    x_train, x_test, y_train, y_test = get_dataset(args.dataset, args.dataset_dim, 800, 200)

    FeatureMap, scaling_factor = HamiltonianEvolutionFeatureMap(args.dataset_dim, args.n_trotter, args.evo_time, args.init_state, args.init_state_seed)
    x_train *= scaling_factor 
    x_test *= scaling_factor

    if args.decimals is not None:
        x_train = np.around(x_train, decimals=args.decimals)
        x_test = np.around(x_test, decimals=args.decimals)

    qkern = get_quantum_kernel(FeatureMap, simulation_method=args.simulation_method, shots=args.shots, batch_size=50)
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
    print(f"Score: {score}")

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
