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
    IQPStyleFeatureMap, 
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
        "--scaling-factor", type = float,
        required = True,
        help = "scale all data by this factor")
    parser.add_argument(
        "--dataset", type = str,
        required = True,
        choices=['fashion-mnist','kmnist','plasticc'],
        help = "dataset to use")
    args = parser.parse_args()
    outpath = Path(args.outpath, f"IQP_dim_{args.dataset_dim}_scale_{args.scaling_factor}.p")
    if outpath.exists():
        print(f"Found already computed at {outpath}, exiting")
        sys.exit()
    else:
        print(f"Using outpath: {outpath}")
    t0 = time.time()

    if args.dataset_dim > 30:
        raise ValueError(f"Dataset dimension {args.dataset_dim} too large; can support no more than 30 qubits")

    x_train, x_test, y_train, y_test = get_dataset(args.dataset, args.dataset_dim, 800, 200)

    x_train *= args.scaling_factor 
    x_test *= args.scaling_factor

    FeatureMap = IQPStyleFeatureMap(args.dataset_dim)
    qkern = get_quantum_kernel(FeatureMap)
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
