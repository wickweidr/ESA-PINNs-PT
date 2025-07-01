# ESA-PINNs-PT

Comment on "Mix-training physics-informed neural networks for the rogue waves of nonlinear Schrödinger equation" [Chaos, Solitons and Fractals 164 (2022) 112712]}.

We propose enhanced self-adaptive PINNs with partition training (ESA-PINNs-PT) to extend the work presented in the commented paper, achieving significantly higher accuracy in resolving rogue wave solutions of the Schrödinger equation.

# 1.
For the first-order rogue wave simulation, a relative $L_2$-norm (RL2) error of 3.20e-5 is achieved, which is over an order of magnitude lower than the error reported for the MTPINNs PLUS model (4.94e-4).

The following two videos illustrate the evolution of self-adaptive weights during the training process with Adam optimizer for solving the first-order rogue wave solution of the Schrödinger equation.

![Video](https://github.com/wickweidr/ESA-PINNs-PT/blob/main/sa-sch-anim-u.gif)
![Video](https://github.com/wickweidr/ESA-PINNs-PT/blob/main/sa-sch-anim-v.gif)

8 tests are conducted to validate the robustness of ESA-PINNs-PT with 1 partition by varying the random seed using 10000 residual points, which achieve RL2 error of (3.98 $\pm$ 1.35)e-5.

When training ESA-PINNs-PT with 3 partitions, higher precision is achieved, yielding the RL2 error of (2.48 $\pm$ 0.38)e-5 across 8 random tests.

# 2.
For the second-order rogue wave simulation, training ESA-PINNs-PT with 3 or more partitions is necessary to ensure high precision, as single-partition training fails to achieve sufficient accuracy.

5 partitions and 20000 (or 10000) residual points are employed to address the more complex characteristics of the second-order rogue wave.

8 tests are also conducted to validate the robustness of ESA-PINNs-PT by varying the random seed using 20000 residual points, which achieve RL2 error of (3.87 $\pm$ 1.06)e-5.

# 3.
Software execution:

python main.py (run code)

Software installation:

PyTorch version 2.X; CUDA version 11.X
