# Running the code

Examples:

```
python compute_quantum_kernel_HamEvo.py --outpath /tmp \
                              --dataset-dim 3 \
                              --n-trotter 20 \
                              --evo-time 0.05 \
                              --init-state Haar_random \
                              --init-state-seed 42 \
                              --dataset kmnist
```

```
python compute_quantum_kernel_IQP.py --outpath /tmp \
                              --dataset-dim 3 \
                              --scaling-factor 0.01 \
                              --dataset plasticc 
```

```
python compute_quantum_kernel_reproduce_huang_et_al.py --outpath /tmp \
                              --dataset-dim 3 \
                              --n-trotter 20 \
                              --init-state Haar_random \
                              --evo-time-factor 0.33 \
                              --init-state-seed 42 
```

Example using `gnu-parallel`:

```
evotimes=( 0.001 0.01 0.05 0.1 0.5 1.0 5.0 )

qubits=( 4 7 10 13 16 19 22 25 )

parallel \
    --jobs 6 \
    """
    python compute_quantum_kernel_HamEvo.py --outpath ~/dev/quantum_kernels/data/results/control_evo_time_kmnist \
                                  --dataset-dim {1} \
                                  --n-trotter {2} \
                                  --evo-time {3} \
                                  --init-state Haar_random \
                                  --init-state-seed {4} \
                                  --dataset kmnist
    """ ::: "${qubits[@]}" ::: $(seq 10 15 40) ::: "${evotimes[@]}" ::: $(seq 1 5)
```
