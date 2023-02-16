# %%
# import numpy and optimize
import torch


def gr_desc_fast(par, gr, maxiter=10000, dat_vals=None, lr=[1], tol=1e-5,
                 verbose=False, epoch=10, **kwargs):
    """Perform Quick Gradient Descent

    Args:
        par (Tensor): Starting Dim
        gr (function): The gradient evaluation function
        maxiter (int, optional): Maximum number of iterations. Defaults to 10000.
        dat_vals (Dictionary, optional): Dictionary of additional values to pass to gradient function.
            Defaults to None.
        lr (list, optional): List of learning rate for each iteration. Defaults to [1].
        tol (float, optional): Value gradient must fall below to stop optimisation. Defaults to 1e-5.

    Returns:
        Dictionary: A dictionary of summaries. These include:
            par (Tensor): Final parameter value.
            par_vec (Tensor): Tensor contaning each parameter value. Each row is an iteration.
            gr_vec (Tensor): Tensor containing each gradient eval. Each row is an iteration.
            converged (int): Indicates convergence. 0=Converged 1=Not Converged
            iter(int): Number of iterations before convergence.
    """
    if dat_vals is None:
        dat_vals = {}
    ndim = len(par)
    par_vec = torch.zeros(maxiter, ndim)
    gr_vec = torch.zeros(maxiter, ndim)

    if type(lr) == int or ((type(lr) == list) & len(lr) == 1):
        lr = torch.tensor(lr).repeat(maxiter)

    converge = 1
    for i in range(maxiter):
        lr_temp = lr[i]
        delta = gr(par, **dat_vals, **kwargs)
        # Test exit condition
        if max(abs(delta)) < tol:
            # print("Algorithm Converged")
            converge = 0
            break

        # If no exit then update vectors for param, gradient etc
        par_vec[i, :] = par
        gr_vec[i, :] = delta

        par = par-lr_temp*delta
        if verbose & (i % epoch == 0):
            print(i)

    par_vec = par_vec[range(i), :]
    gr_vec = gr_vec[range(i), :]

    return {"par": par, "par_vec": par_vec, "gr": delta,
            "gr_vec": gr_vec, "converged": converge, "iter": i}
