# ESA-PINNs-PT

Comment on "Mix-training physics-informed neural networks for the rogue waves of nonlinear Schrödinger equation" [Chaos, Solitons and Fractals 164 (2022) 112712]}.

An alternative approach for handling PDEs with sharp gradients is the self-adaptive PINNs (SA-PINNs) method.

We propose enhanced SA-PINNs with partition training (ESA-PINNs-PT) to extend the work presented in the commented paper, achieving significantly higher accuracy in resolving rogue wave solutions of the Schrödinger equation.

1.
For the first-order rogue wave simulation, a relative $L_2$-norm (RL2) error of $3.20 \mathrm{e}-5$ is achieved, which is over an order of magnitude lower than the error reported for the MTPINNs PLUS model ($4.94\times 10^{-4}$).

8 tests are conducted to validate the robustness of ESA-PINNs-PT by varying the random seed using 10000 residual points, which achieve RL2 error of $(3.98 \pm 1.35) \mathrm{e}-5$.

The following two videos illustrate the evolution of self-adaptive weights during the training process with Adam optimizer for solving the first-order rogue wave solution of the Schrödinger equation.

![Video](https://github.com/wickweidr/SchrodingerEq/blob/main/sa-sch-anim-u.gif)
![Video](https://github.com/wickweidr/SchrodingerEq/blob/main/sa-sch-anim-v.gif)

2.
The second-order rogue wave solution of the Schrödinger equation, as presented in [Phys. Rev. E 85, 026607 (2012)], is accurately resolved using the enhanced SA-PINNs combined with a time-domain decomposition method. The corresponding contour plots are provided in sch12-2nd-t2x3-jet.pdf.

For the second-order rogue wave simulation, the time domain is partitioned into five uniform subdomains, with at least three subdomains required to achieve high-accuracy predictions.
Training a single neural network over the entire domain fails to yield correct results, indicating the necessity of domain decomposition. 

3.
Software execution:

python main.py (run code)

Software installation:

PyTorch version 2.X; CUDA version 11.X
