# %%
# Modules
import torch
from torch import nn
import scipy.optimize as opt
import sys
sys.path.append('..')

# Script Functions
from functions.objective_funcs_torch import adjusted_logreg, gr_adj_logreg  # noqa: E402
from functions.objective_funcs_torch import gr_kliep_miss_f, nkliep_theta_approx_fast, neg_kliep_miss_f  # noqa: E402
from functions.gradient_desc_torch import gr_desc_fast  # noqa: E402


# %%
# Torch Version
# Proper way
class KLIEP_Model(nn.Module):
    """Custom Pytorch model for gradient optimization.
    """

    def __init__(self, weights):

        super().__init__()
        # initialize weights with random numbers
        # make weights torch parameters
        self.weights = nn.Parameter(weights)

    def forward(self, **dat_vals):
        """Implement function to be optimised. This this case it is KLIEP
        """
        return neg_kliep_miss_f(self.weights, **dat_vals)


def training_loop(model, optimizer, maxiter=1000, tol=1e-4, **kwargs):
    "Training loop for torch model."
    losses = []
    for i in range(maxiter):
        loss = model(**kwargs)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        losses.append(loss)
    return losses


def training_loop_bfgs(model, optimizer, maxiter=1000, tol=1e-4, **kwargs):
    "Training loop for torch model."
    losses = []
    for i in range(maxiter):
        optimizer.zero_grad()
        loss = model(**kwargs)
        loss.backward()
        if torch.max(torch.abs(model.weights.grad)) < tol:
            break
        optimizer.step(lambda: model(**kwargs))
        losses.append(loss)
    return losses


# Create wrapper for getting the optimal theta values for each method from data
def kliep_miss_wrap_grad(gen_dat, norm_fl=False, f=None, maxiter=1000, reg=0., **kwargs):
    """Performs standard M-KLIEP by Gradient descent

    Args:
        gen_dat (Dictionary): Dictionary of data summaries to be passed on to neg_kliep_miss_f
        norm_fl (bool, optional): Whether to calculate normalisation term. Defaults to False.
        f (function, optional): f function to use in exponential form for M-KLIEP. Defaults to None.
        maxiter (int, optional): Maximum number of iterations. Defaults to 1000.
        reg (float, optional): optional L2 regularisation. Defaults to 0.

    Returns:
        Dictionary: Contains the following items
            par (Tensor) - The estimated parameter
            gr (Tensor) - The gradient at the final opimisation step
            converged (Int) - Returns 0 if the method converged
            iter (Int) - The number of iterations run
            Norm_term (Float) - The normalisation term
            r (Function) -  The density ratio estimate prodcued.
    """
    # Trim dictionary to get relvant quantites
    dat = {k: gen_dat[k] for k in ("f_val_plus", "f_val_minus", "n_plus", "n_minus",
                                   "varphi_val_plus", "varphi_val_minus")}
    dat["reg"] = reg
    dim = dat["f_val_plus"].shape[1]
    # Get optim value
    out_optim = gr_desc_fast(
        par=torch.zeros(dim), gr=gr_kliep_miss_f, dat_vals=dat,
        maxiter=maxiter, **kwargs)
    theta_opt = out_optim["par"]
    out_dict = {key: out_optim[key] for key in
                ["par", "gr", "converged", "iter"]}

    if norm_fl:
        Norm_term = nkliep_theta_approx_fast(
            theta=theta_opt, f_val_minus=dat["f_val_minus"],
            varphi_val_minus=dat["varphi_val_minus"],
            n_minus=dat["n_minus"])
        out_dict["Norm_term"] = Norm_term
        if f is not None:
            def r(x):
                return torch.exp(f(x)@theta_opt)/Norm_term
            out_dict["r"] = r
    return out_dict


# Create wrapper for getting the optimal theta values for each method from data
def kliep_miss_wrap_scalar(gen_dat, norm_fl=False, f=None, maxiter=1000, reg=0., **kwargs):
    """Performs M-KLIEP for 1-dim theta

    Args:
        gen_dat (Dictionary): Dictionary of data summaries to be passed on to neg_kliep_miss_f
        norm_fl (bool, optional): Whether to calculate normalisation term. Defaults to False.
        f (function, optional): f function to use in exponential form for M-KLIEP. Defaults to None.
        maxiter (int, optional): Maximum number of iterations. Defaults to 1000.
        reg (float, optional): optional L2 regularisation. Defaults to 0..

    Returns:
        Dictionary: Contains the following items
            par (Tensor) - The estimated parameter
            gr (Tensor) - The evaluation at the final step
            converged (Int) - Returns 0 if the method converged
            iter (Int) - The number of iterations run
            Norm_term (Float) - The normalisation term
            r (Function) -  The density ratio estimate produced.
    """
    # Trim dictionary to get relvant quantites
    dat_tup = [gen_dat[k] for k in ("f_val_plus", "f_val_minus", "n_plus", "n_minus",
                                    "varphi_val_plus", "varphi_val_minus")]
    dat_tup.append(reg)
    dat_tup = tuple(dat_tup)
    options = {"maxiter": maxiter}
    # dim = gen_dat["f_val_plus"].shape[1]
    # Get optim value
    out_optim = opt.minimize_scalar(
        fun=neg_kliep_miss_f,
        args=dat_tup,
        method="bounded",
        bounds=(-10., 10.),
        options=options,
    )
    theta_opt = out_optim["x"]
    if type(theta_opt) != torch.Tensor:
        theta_opt = torch.tensor(theta_opt).float()
    else:
        theta_opt = theta_opt.float()
    theta_opt = theta_opt.reshape(-1)
    out_dict = {
        "par": theta_opt,
        "gr": out_optim["fun"],
        "converged": 1-out_optim["success"], "iter": out_optim["nfev"]}

    if norm_fl:
        Norm_term = nkliep_theta_approx_fast(
            theta=theta_opt, f_val_minus=gen_dat["f_val_minus"],
            varphi_val_minus=gen_dat["varphi_val_minus"],
            n_minus=gen_dat["n_minus"])
        out_dict["Norm_term"] = Norm_term
        if f is not None:
            def r(x):
                return torch.exp(f(x)@theta_opt)/Norm_term
            out_dict["r"] = r
    return out_dict


def kliep_miss_wrap_adam(gen_dat, norm_fl=False, f=None, maxiter=1000, reg=0., **kwargs):
    """Performs standard M-KLIEP using AdAM for the optimisation

    Args:
    gen_dat (Dictionary): Dictionary of data summaries to be passed on to neg_kliep_miss_f
        norm_fl (bool, optional): Whether to calculate normalisation term. Defaults to False.
        f (function, optional): f function to use in exponential form for M-KLIEP. Defaults to None.
        maxiter (int, optional): Maximum number of iterations. Defaults to 1000.
        reg (float, optional): optional L2 regularisation. Defaults to 0.

    Returns:
        Dictionary: Contains the following items:
            par (Tensor) - The estimated parameter
            gr (Tensor) - The gradient at the final opimisation step
            converged (Int) - Returns 0 if the method converged
            iter (Int) - The number of iterations run
            Norm_term (Float) - The normalisation term
            r (Function) -  The density ratio estimate produced.
    """
    # Trim dictionary to get relvant quantites
    dat = {k: gen_dat[k] for k in ("f_val_plus", "f_val_minus", "n_plus", "n_minus",
                                   "varphi_val_plus", "varphi_val_minus")}
    dat["reg"] = reg
    dim = dat["f_val_plus"].shape[1]
    m = KLIEP_Model(weights=torch.zeros(dim))
    # Instantiate optimizer
    opt = torch.optim.Adam(m.parameters(), lr=0.1)
    # Get optim value
    training_loop(m, opt, **dat, maxiter=maxiter)
    theta_opt = m.weights.detach()
    out_dict = {"par": theta_opt, "gr": m.weights.grad,
                "converged": 0, "iter": maxiter}

    if norm_fl:
        Norm_term = nkliep_theta_approx_fast(
            theta=theta_opt, f_val_minus=dat["f_val_minus"],
            varphi_val_minus=dat["varphi_val_minus"],
            n_minus=dat["n_minus"])
        out_dict["Norm_term"] = Norm_term
        if f is not None:
            def r(x):
                return torch.exp(f(x)@theta_opt)/Norm_term
            out_dict["r"] = r
    return out_dict


def kliep_miss_wrap_bfgs(gen_dat, norm_fl=False, f=None, maxiter=1000, reg=0.,
                         lr=1, tol=1e-4, **kwargs):
    """Performs standard M-KLIEP using LBFGS for the optimisation

    Args:
    gen_dat (Dictionary): Dictionary of data summaries to be passed on to neg_kliep_miss_f
        norm_fl (bool, optional): Whether to calculate normalisation term. Defaults to False.
        f (function, optional): f function to use in exponential form for M-KLIEP. Defaults to None.
        maxiter (int, optional): Maximum number of iterations. Defaults to 1000.
        reg (float, optional): optional L2 regularisation. Defaults to 0.

    Returns:
        Dictionary: Contains the following items:
            par (Tensor) - The estimated parameter
            gr (Tensor) - The gradient at the final opimisation step
            converged (Int) - Returns 0 if the method converged
            iter (Int) - The number of iterations run
            Norm_term (Float) - The normalisation term
            r (Function) -  The density ratio estimate produced.
    """
    # Trim dictionary to get relvant quantites
    dat = {k: gen_dat[k] for k in ("f_val_plus", "f_val_minus", "n_plus", "n_minus",
                                   "varphi_val_plus", "varphi_val_minus")}
    dat["reg"] = reg
    dim = dat["f_val_plus"].shape[1]
    m = KLIEP_Model(weights=torch.zeros(dim))
    # Instantiate optimizer
    opt = torch.optim.LBFGS(
        m.parameters(), lr=lr, line_search_fn='strong_wolfe')
    # Get optim value
    training_loop_bfgs(m, opt, **dat, maxiter=maxiter, tol=tol)
    theta_opt = m.weights.detach()
    out_dict = {"par": theta_opt, "gr": m.weights.grad,
                "converged": 0, "iter": maxiter}

    if norm_fl:
        Norm_term = nkliep_theta_approx_fast(
            theta=theta_opt, f_val_minus=dat["f_val_minus"],
            varphi_val_minus=dat["varphi_val_minus"],
            n_minus=dat["n_minus"])
        out_dict["Norm_term"] = Norm_term
        if f is not None:
            def r(x):
                return torch.exp(f(x)@theta_opt)/Norm_term
            out_dict["r"] = r
    return out_dict


opt_dict = {"BFGS": kliep_miss_wrap_bfgs, "ADAM": kliep_miss_wrap_adam,
            "GRADIENT DESCENT": kliep_miss_wrap_grad, "SCALAR": kliep_miss_wrap_scalar}
opt_types = opt_dict.keys()


def kliep_miss_wrap(gen_dat, norm_fl=False, f=None, maxiter=1000, reg=0.,
                    opt_type="Gradient Descent", **kwargs):
    """Performs  M-KLIEP using specified optimiser

    Args:
        gen_dat (Dictionary): Dictionary of data summaries to be passed on to neg_kliep_miss_f
        norm_fl (bool, optional): Whether to calculate normalisation term. Defaults to False.
        f (function, optional): f function to use in exponential form for M-KLIEP. Defaults to None.
        maxiter (int, optional): Maximum number of iterations. Defaults to 1000.
        reg (float, optional): optional L2 regularisation. Defaults to 0..
        opt_type (str, optional): Optiomisation to perform.
                                  One of: "Gradient Descent", "Adam", "Scalar", "BFGS".
                                  Defaults to "Gradient Descent".


    Returns:
        Dictionary: Contains the following items
            par (Tensor) - The estimated parameter
            gr (Tensor) - The evaluation at the final step
            converged (Int) - Returns 0 if the method converged
            iter (Int) - The number of iterations run
            Norm_term (Float) - The normalisation term
            r (Function) -  The density ratio estimate produced.
    """
    if opt_type.upper() in opt_types:
        out = opt_dict[opt_type.upper()](
            gen_dat, norm_fl, f=f, maxiter=maxiter, reg=reg, **kwargs)
    else:
        raise Exception(
            "Invalid opt_type. Options are: \"Gradient Descent\", \"ADAM\", \"Scalar\", \"BFGS\"")
    return out


def kliep_naive_wrap(gen_dat, norm_fl=False, f=None, maxiter=1000, reg=0.,
                     opt_type="Gradient Descent", **kwargs):
    """Performs  CC-KLIEP using specified optimiser

    Args:
        gen_dat (Dictionary): Dictionary of data summaries to be passed on to neg_kliep_miss_f
        norm_fl (bool, optional): Whether to calculate normalisation term. Defaults to False.
        f (function, optional): f function to use in exponential form for M-KLIEP. Defaults to None.
        maxiter (int, optional): Maximum number of iterations. Defaults to 1000.
        reg (float, optional): optional L2 regularisation. Defaults to 0..
        opt_type (str, optional): Optiomisation to perform.
                                  One of: "Gradient Descent", "Adam", "Scalar", "BFGS".
                                  Defaults to "Gradient Descent".


    Returns:
        Dictionary: Contains the following items
            par (Tensor) - The estimated parameter
            gr (Tensor) - The evaluation at the final step
            converged (Int) - Returns 0 if the method converged
            iter (Int) - The number of iterations run
            Norm_term (Float) - The normalisation term
            r (Function) -  The density ratio estimate produced.
    """
    # Trim dictionary to get relvant quantites
    dat = {k: gen_dat[k] for k in ("f_val_plus", "f_val_minus")}
    dat["n_plus"] = gen_dat["n_plus"]-gen_dat["nmiss_plus"]
    dat["n_minus"] = gen_dat["n_minus"]-gen_dat["nmiss_minus"]
    dat["varphi_val_plus"] = None
    dat["varphi_val_minus"] = None
    out = kliep_miss_wrap(dat, norm_fl, f, maxiter, reg, opt_type, **kwargs)
    return out


def kliep_multi_dim_imp_wrap(gen_dat, norm_fl=False, f=None, maxiter=1000,
                             opt_type="Gradient Descent", **kwargs):
    """Performs  KLIEP using imputed (multi-dimensional) data with given optimiser.

    Args:
        gen_dat (Dictionary): Dictionary of data summaries to be passed on to neg_kliep_miss_f
        norm_fl (bool, optional): Whether to calculate normalisation term. Defaults to False.
        f (function, optional): f function to use in exponential form for M-KLIEP. Defaults to None.
        maxiter (int, optional): Maximum number of iterations. Defaults to 1000.
        reg (float, optional): optional L2 regularisation. Defaults to 0..
        opt_type (str, optional): Optiomisation to perform.
                                  One of: "Gradient Descent", "Adam", "Scalar", "BFGS".
                                  Defaults to "Gradient Descent".


    Returns:
        Dictionary: Contains the following items
            par (Tensor) - The estimated parameter
            gr (Tensor) - The evaluation at the final step
            converged (Int) - Returns 0 if the method converged
            iter (Int) - The number of iterations run
            Norm_term (Float) - The normalisation term
            r (Function) -  The density ratio estimate produced.
    """
    gen_dat["f_val_plus"] = torch.cat(gen_dat["f_imp_plus"], dim=1)
    gen_dat["f_val_minus"] = torch.cat(gen_dat["f_imp_minus"], dim=1)
    gen_dat["n_plus"] = gen_dat["f_val_plus"].shape[0]
    gen_dat["n_minus"] = gen_dat["f_val_minus"].shape[0]
    gen_dat["varphi_val_plus"] = None
    gen_dat["varphi_val_minus"] = None

    if opt_type.upper() in opt_types:
        out = opt_dict[opt_type.upper()](
            gen_dat, norm_fl, f=f, maxiter=maxiter, **kwargs)
    else:
        raise Exception(
            "Invalid opt_type. Options are: \"Gradient Descent\", \"ADAM\", \"Scalar\", \"BFGS\"")

    return out


def kliep_multi_dim_sep_wrap(gen_dat, norm_fl=False, f=None, maxiter=1000, reg=0.,
                             opt_type="Gradient Descent", **kwargs):
    """Performs M-KLIEP separately over each dimension with given optimiser

    Args:
        gen_dat_multi (Dictionary): Output from get_dat_vals_multidim
        norm_fl (bool, optional): Whether to compute normalisation term. Defaults to False.
        f (function, optional): The f to use in log-linear parametric form for EACH dimension. Defaults to None.
        opt_type (str, optional): Which optimisation type to use.
                                  Options are:, "Gradient Descent", "BFGS", "Adam", and "Scalar".
                                  Defaults to "Gradient Descent".

    Returns:
        Dictionary: Contains the following items:
            par (Tensor) - The estimated parameter, each column is a data dimension
            gr (Tensor) - The gradient at the final opimisation step, each column is a data dimension
            converged (List) - Returns 0 if the method converged
            iter (List) - The number of iterations run
            Norm_term (List) - The normalisation term
            r (function) -  The density ratio estimate produced.
    """
    # Trim dictionary to get relvant quantites
    dat = {k: gen_dat[k] for k in
           ("f_val_plus", "f_val_minus", "nmiss_plus", "nmiss_minus",
           "varphi_val_plus", "varphi_val_minus")}
    dim = len(dat["f_val_plus"])
    per_dim = dat["f_val_plus"][0].shape[1]
    out_dict = {
        "par": torch.zeros((per_dim, dim)), "gr": torch.zeros((per_dim, dim)),
        "converged": [], "iter": []}
    if f is not None:
        out_dict["list_r"] = []
    if norm_fl:
        out_dict["Norm_term"] = []
    for j in range(dim):
        # Set up data
        temp_dat = {key: value[j] for key, value in dat.items()}
        temp_dat["n_plus"] = temp_dat["nmiss_plus"] + \
            temp_dat["f_val_plus"].shape[0]
        temp_dat["n_minus"] = temp_dat["nmiss_minus"] + \
            temp_dat["f_val_minus"].shape[0]
        temp_dat["reg"] = reg

        del temp_dat["nmiss_minus"]
        del temp_dat["nmiss_plus"]

        temp_out = kliep_miss_wrap(
            temp_dat, norm_fl, f, maxiter, reg, opt_type, **kwargs)

        # Write out values
        out_dict["par"][:, j] = temp_out["par"]
        out_dict["gr"][:, j] = temp_out["gr"]
        out_dict["converged"].append(temp_out["converged"])
        out_dict["iter"].append(temp_out["iter"])
        if f is not None:
            out_dict["list_r"].append(temp_out["r"])
        if norm_fl:
            out_dict["Norm_term"].append(temp_out["Norm_term"])

    # Create density ratio estimate
    if f is not None:
        def r(x):
            vals = torch.stack([
                r_sub(x[:, j:j+1]) for j, r_sub in enumerate(out_dict["list_r"])
            ], dim=1)
            return torch.prod(vals, dim=1)

        out_dict["r"] = r
    return out_dict


def kliep_multi_dim_naive_sep_wrap(gen_dat, norm_fl=False, f=None, reg=0.,
                                   opt_type="Gradient Descent", **kwargs):
    """Performs CC-KLIEP separately over each dimension

    Args:
        gen_dat_multi (Dictionary): Output from get_dat_vals_multidim
        norm_fl (bool, optional): Whether to compute normalisation term. Defaults to False.
        f (function, optional): The f to use in log-linear parametric form for EACH dimension. Defaults to None.
        reg (float, optional): The L" regularisation to use in fitting. Defaults to 0.
        opt_type (str, optional): Which optimisation type to use.
        Options are:, "Gradient Descent", "BFGS", "Adam", and "Scalar". Defaults to "Gradient Descent".

    Returns:
        Dictionary: Contains the following items:
            par (Tensor) - The estimated parameter, each column is a data dimension
            gr (Tensor) - The gradient at the final opimisation step, each column is a data dimension
            converged (List) - Returns 0 if the method converged
            iter (List) - The number of iterations run
            Norm_term (List) - The normalisation term
            r (function) -  The density ratio estimate produced.
    """

    d = len(gen_dat["f_val_plus"])
    gen_dat_trim = {
        "f_val_plus": gen_dat["f_val_plus"], "f_val_minus": gen_dat["f_val_minus"],
        "varphi_val_plus": [None]*d, "varphi_val_minus": [None]*d,
        "nmiss_plus": [0]*d, "nmiss_minus": [0]*d}
    out = kliep_multi_dim_sep_wrap(
        gen_dat_trim, norm_fl=norm_fl, f=f, reg=reg, opt_type=opt_type, **kwargs)
    return out


def kliep_multi_dim_sep_imp_wrap(gen_dat, norm_fl=False, f=None, maxiter=1000,
                                 opt_type="Gradient Descent", **kwargs):
    """Performs  KLIEP separately over each dimension using imputed (multi-dimensional) data with given optimiser.

    Args:
        gen_dat (Dictionary): Dictionary of data summaries to be passed on to neg_kliep_miss_f
        norm_fl (bool, optional): Whether to calculate normalisation term. Defaults to False.
        f (function, optional): f function to use in exponential form for M-KLIEP. Defaults to None.
        maxiter (int, optional): Maximum number of iterations. Defaults to 1000.
        reg (float, optional): optional L2 regularisation. Defaults to 0..
        opt_type (str, optional): Optiomisation to perform.
                                  One of: "Gradient Descent", "Adam", "Scalar", "BFGS".
                                  Defaults to "Gradient Descent".


    Returns:
        Dictionary: Contains the following items
            par (Tensor) - The estimated parameter
            gr (Tensor) - The evaluation at the final step
            converged (Int) - Returns 0 if the method converged
            iter (Int) - The number of iterations run
            Norm_term (Float) - The normalisation term
            r (Function) -  The density ratio estimate produced.
    """
    d = len(gen_dat["f_imp_plus"])
    gen_dat_trim = {
        "f_val_plus": gen_dat["f_imp_plus"],
        "f_val_minus": gen_dat["f_imp_minus"],
        "nmiss_plus": [0]*d, "nmiss_minus": [0]*d,
        "varphi_val_plus": [None]*d, "varphi_val_minus": [None]*d
    }
    out = kliep_multi_dim_sep_wrap(
        gen_dat=gen_dat_trim, norm_fl=norm_fl, f=f, maxiter=maxiter,
        opt_type=opt_type, **kwargs)
    return out


def adj_logreg_wrap(x, y, **kwargs):
    dim = 3
    dat_tup = (x, y)
    # Get optim value
    out_optim = opt.minimize(
        fun=adjusted_logreg,
        x0=torch.zeros(dim),
        jac=gr_adj_logreg, args=dat_tup, method="BFGS"
    )
    return {"par": out_optim["x"], "converged": 1-out_optim["success"],
            "iter": out_optim["nit"]}

# %%
