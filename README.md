# PINN Diffusion Tutorial

A PyTorch tutorial comparing a purely data-driven neural network with a physics-informed neural network (PINN) for the 1D diffusion equation.

The goal is to show why adding physical laws to the training objective can make neural network predictions more physically meaningful, especially when only sparse and noisy data are available.

## Overview

This repository contains a Jupyter notebook that walks through a simple diffusion example step by step.

We study the 1D diffusion equation

```math
u_t = \alpha u_{xx}
```

where `u(t,x)` can represent temperature, concentration, or density, and `alpha` is the diffusion coefficient.

The notebook compares two approaches:

1. A data-only neural network trained only on sparse observations.
2. A physics-informed neural network trained on sparse observations plus the governing PDE.

The data-only model can fit the observed data points but may violate the diffusion physics between and beyond those points. The PINN is trained to satisfy both the data and the PDE residual,

```math
r(t,x) = u_t - \alpha u_{xx}
```

which encourages physically feasible predictions across the full space-time domain.

## Main Idea

We generate synthetic data from an analytic Gaussian solution of the diffusion equation. The initial condition is a Gaussian pulse centered at `x = 0.5`. Over time, diffusion causes the pulse to become wider and lower.

The exact solution is

```math
u(x,t) =
\frac{\sigma}{\sqrt{\sigma^2 + 2\alpha t}}
\exp\left(
-\frac{(x-\mu)^2}{2(\sigma^2 + 2\alpha t)}
\right)
```

We then train two models on sparse noisy observations, mostly from early times.

The data-only neural network minimizes only

```math
\mathcal{L}_{data}
=
\frac{1}{N}
\sum_i
\left(
u_\theta(t_i,x_i) - u_i
\right)^2
```

The PINN minimizes a combined loss,

```math
\mathcal{L}
=
\lambda_{data}\mathcal{L}_{data}
+
\lambda_{IC}\mathcal{L}_{IC}
+
\lambda_{BC}\mathcal{L}_{BC}
+
\lambda_{PDE}\mathcal{L}_{PDE}
```

where

```math
\mathcal{L}_{PDE}
=
\frac{1}{N}
\sum_j
\left(
u_t(t_j,x_j) - \alpha u_{xx}(t_j,x_j)
\right)^2
```

## What the Tutorial Shows

The notebook demonstrates that:

- a neural network can fit sparse data while still violating the underlying physics,
- a PINN can use the governing PDE to improve physical consistency,
- PDE residuals provide a useful diagnostic for physical feasibility,
- the data-only model may diffuse too slowly or produce physically questionable behavior,
- the PINN better captures the expected smoothing and decay of the diffusion process.
