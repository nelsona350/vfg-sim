# VFG With Integrated PN Logic

This document explains the major calculations in `vfg_with_ipn.py`. The script simulates a vehicle and target in a North-East-Down (NED) frame, writes the output in East-North-Up (ENU), and switches the vehicle from waypoint-following vector-field guidance (VFG) to two-angle integrated proportional navigation (PN) near terminal range.

## Coordinate Frames

The simulation state is propagated in NED coordinates:

$$
\mathbf{p}_{NED}=\begin{bmatrix}N & E & D\end{bmatrix}^T
$$

Output rows are converted to ENU with:

$$
\mathbf{p}_{ENU}=\begin{bmatrix}E & N & -D\end{bmatrix}^T
$$

The same component swap and sign change is applied to velocities and waypoints.

## Constant-Heading Target Initialization

Before full guidance updates, the target is initialized along the first leg from its initial position to its first waypoint:

$$
\mathbf{r}=\mathbf{w}_0-\mathbf{p}_0
$$

$$
\hat{\mathbf{u}}=\frac{\mathbf{r}}{\lVert\mathbf{r}\rVert}
$$

$$
\mathbf{v}=V_d\hat{\mathbf{u}}
$$

where $V_d$ is the desired flight speed. A near-zero leg length is rejected because it cannot define a heading.

## Launch-Time Synchronization

The vehicle launch time is chosen so the vehicle reaches its final waypoint when the target reaches point of closest approach (POCA) to that waypoint.

For a target with initial position $\mathbf{p}_t$, velocity $\mathbf{v}_t$, and vehicle final waypoint $\mathbf{w}_f$, the target POCA time is:

$$
t_{POCA}=\max\left(0,\frac{(\mathbf{w}_f-\mathbf{p}_t)\cdot\mathbf{v}_t}{\mathbf{v}_t\cdot\mathbf{v}_t}\right)
$$

The vehicle route time is the total waypoint path length divided by desired speed:

$$
t_v=\frac{\sum_{i=0}^{n-1}\lVert\mathbf{q}_{i+1}-\mathbf{q}_i\rVert}{V_v}
$$

where $\mathbf{q}_0$ is the vehicle initial position and subsequent $\mathbf{q}$ values are waypoints. The synchronized launch time from simulation time $t$ is:

$$
t_L=t+t_{POCA}-t_v
$$

## Vehicle Waypoint Updates

The vehicle penultimate waypoint is rotated around the final waypoint so the final approach is anti-parallel to the target path while preserving final-leg length $L_f$:

$$
\mathbf{w}_{n-1}'=\mathbf{w}_n+L_f\frac{\mathbf{v}_t}{\lVert\mathbf{v}_t\rVert}
$$

After the vehicle reaches the penultimate waypoint, the final waypoint may be moved to a predicted intercept point. Intercept time $t$ solves:

$$
\lVert\mathbf{p}_t+\mathbf{v}_t t-\mathbf{p}_v\rVert^2=(V_v t)^2
$$

which expands into:

$$
(\mathbf{v}_t\cdot\mathbf{v}_t-V_v^2)t^2+2(\mathbf{p}_t-\mathbf{p}_v)\cdot\mathbf{v}_t\,t+\lVert\mathbf{p}_t-\mathbf{p}_v\rVert^2=0
$$

The earliest nonnegative root is used. If no real root exists, the script falls back to the target's closest approach to the vehicle position.

## Vector-Field Guidance

For a route segment from previous waypoint $\mathbf{w}_{i-1}$ to current waypoint $\mathbf{w}_i$:

$$
\mathbf{s}=\mathbf{w}_i-\mathbf{w}_{i-1},\qquad
\hat{\mathbf{s}}=\frac{\mathbf{s}}{\lVert\mathbf{s}\rVert}
$$

The along-track distance from the previous waypoint is:

$$
d_a=(\mathbf{p}-\mathbf{w}_{i-1})\cdot\hat{\mathbf{s}}
$$

The finite-segment projection clamps this distance into $[0,\lVert\mathbf{s}\rVert]$:

$$
\mathbf{p}_c=\mathbf{w}_{i-1}+\operatorname{clip}(d_a,0,\lVert\mathbf{s}\rVert)\hat{\mathbf{s}}
$$

The cross-track error vector removes any component parallel to the path:

$$
\mathbf{e}_c=(\mathbf{p}-\mathbf{p}_c)-\left((\mathbf{p}-\mathbf{p}_c)\cdot\hat{\mathbf{s}}\right)\hat{\mathbf{s}}
$$

A smooth intercept angle is scheduled with:

$$
\alpha=\alpha_{max}\tanh\left(\frac{\lVert\mathbf{e}_c\rVert}{\tau V_d}\right)
$$

where $\tau$ is the convergence time constant. The commanded direction combines along-track motion and cross-track correction:

$$
\hat{\mathbf{g}}=\cos(\alpha)\hat{\mathbf{s}}-\sin(\alpha)\hat{\mathbf{e}}_c
$$

$$
\mathbf{v}_{cmd}=V_d\hat{\mathbf{g}}
$$

A waypoint is considered crossed when:

$$
d_a \ge \lVert\mathbf{s}\rVert - \epsilon
$$

where $\epsilon$ is the waypoint-plane tolerance.

## Two-Angle Integrated Proportional Navigation

Terminal guidance uses independent horizontal and vertical line-of-sight (LOS) angles. From relative NED position $\mathbf{r}=\mathbf{p}_t-\mathbf{p}_v$:

$$
\psi_{LOS}=\operatorname{atan2}(E,N)
$$

$$
\theta_{LOS}=\operatorname{atan2}(-D,\sqrt{N^2+E^2})
$$

The guidance law integrates LOS angle changes into commanded heading $\chi$ and flight-path angle $\gamma$:

$$
\chi_k=\operatorname{wrap}_{\pi}\left(\chi_{k-1}+N_h\operatorname{wrap}_{\pi}(\psi_{LOS,k}-\psi_{LOS,k-1})\right)
$$

$$
\gamma_k=\operatorname{clip}\left(\operatorname{wrap}_{\pi}\left(\gamma_{k-1}+N_v\operatorname{wrap}_{\pi}(\theta_{LOS,k}-\theta_{LOS,k-1})\right),-\frac{\pi}{2},\frac{\pi}{2}\right)
$$

The commanded velocity is reconstructed at fixed speed $V_d$:

$$
V_h=V_d\cos(\gamma_k)
$$

$$
\mathbf{v}_{cmd}=\begin{bmatrix}
V_h\cos(\chi_k)\\
V_h\sin(\chi_k)\\
-V_d\sin(\gamma_k)
\end{bmatrix}
$$

## POCA Detection During Simulation

For each integration step, relative position and velocity are:

$$
\mathbf{r}=\mathbf{p}_t-\mathbf{p}_v,\qquad
\mathbf{v}_r=\mathbf{v}_t-\mathbf{v}_v
$$

The within-step closest-approach time is:

$$
t^*=-\frac{\mathbf{r}\cdot\mathbf{v}_r}{\mathbf{v}_r\cdot\mathbf{v}_r}
$$

If $0\le t^*\le\Delta t$, the simulation advances only to $t^*$ and stops at POCA. Otherwise, it advances a normal time step.

## Outputs

The script writes two output products:

1. Translational state history with vehicle state, target state, dynamic waypoints, range, and integrated-PN activation state.
2. Prelaunch target propagations sampled at 10 Hz for each launch-time/waypoint recomputation.
