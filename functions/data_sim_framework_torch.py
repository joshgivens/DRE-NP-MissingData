# %%
import torch
from torch import distributions
import pandas as pa
import gc
import sys
sys.path.append('..')

from functions.objective_funcs_torch import get_dat_vals_impute  # noqa:E402
from functions.pipeline_funcs import corrupt_func  # noqa:E402
from functions.np_classifier_torch import cutoff_bin, power_alpha_calc  # noqa: E402
unif = distributions.Uniform(0, 1)
# %%


# Simulation generation framework
def create_data_gen(z_plus_gen, z_minus_gen, nsiml, estimators,
                    miss_func_plus=None, miss_func_minus=None,
                    f=lambda x: x, impute=False, partial_miss=False,
                    dat_vals_fun=get_dat_vals_impute, **kwargs):
    """Performs simulated data generation and specified DRE approaches

    Args:
        z_plus_gen (List): List of data generating functions to loop through
        z_minus_gen (List): List of negative class data generating functions to loop through
        nsiml (int): Number of simulations per scenario
        estimators (List): List of DRE estimators as functions (from estimators.torch)
        miss_func_plus (List, optional): List of positive class missing functions. Defaults to None.
        miss_func_minus (List, optional): List of negative class missing functions. Defaults to None.
        f (Function, optional): Function to apply in KLIEP standard form. Defaults to lambdax:x.
        impute (bool, optional): Whether imputed version of data should be produced. Defaults to True.
        dat_vals_fun (Function, optional): Function which summarises data for use with esimtators. Defaults to get_dat_vals_impute.

    Returns:
        Pandas.dataframe: A dataframe of the estimated parameters from each run and associated convergence diagnostics
    """

    # Get number of outer iterations
    nouter = len(z_plus_gen)
    counter = 0
    optim_list = []

    # For each different scenario
    for i in range(nouter):
        if miss_func_plus is not None:
            varphi_plus = miss_func_plus[i]
        else:
            varphi_plus = None

        if miss_func_minus is not None:
            varphi_minus = miss_func_minus[i]
        else:
            varphi_minus = None

        # For each simulation
        for j in range(nsiml):
            # Create "true" data
            z_plus = z_plus_gen[i]()
            z_minus = z_minus_gen[i]()
            n_plus = z_plus.shape[0]
            n_minus = z_minus.shape[0]
            dim = z_minus.shape[1]

            x_minus = z_minus.detach()
            x_plus = z_plus.detach()

            if partial_miss:
                x_plus = corrupt_func(x_plus, varphi_plus)
                x_minus = corrupt_func(x_minus, varphi_minus)

            else:
                # If required create corrupted + data
                if varphi_plus is not None:
                    # Create corrupted data
                    u_plus = unif.sample((n_plus,))
                    x_plus[
                        u_plus < varphi_plus[i](z_plus), :
                    ] = torch.nan

                # If required create corrupted - data
                if varphi_minus is not None:
                    # Create corrupted data
                    u_minus = unif.sample((n_minus))
                    x_minus[
                        u_minus < varphi_minus(z_minus), :
                    ] = torch.nan

            # remove perfect data
            del z_plus, z_minus

            # Compute data summaries
            dat = dat_vals_fun(
                x_plus=x_plus, x_minus=x_minus, f=f,
                varphi_plus=varphi_plus,
                varphi_minus=varphi_minus, impute=impute
            )

            # remove imperfect data
            del x_plus, x_minus

            # Find proportion of missing data
            plus_miss_prop = dat["nmiss_plus"]/dat["n_plus"]
            minus_miss_prop = dat["nmiss_minus"]/dat["n_minus"]

            for func_key, function in estimators.items():
                out_optim = function(dat, norm_fl=True, **kwargs)
                optim_list.append(
                    out_optim["par"].tolist()
                    + out_optim["gr"].tolist()
                    + [i, j, out_optim["converged"],
                       float(out_optim["Norm_term"]), plus_miss_prop,
                       minus_miss_prop, out_optim["iter"], func_key]
                )

            # Remove data summaries
            del dat, out_optim
            gc.collect()

            # Update iteration counter
            counter += 1

    # Append useful values to list
    colnames = (["Param" + str(i) for i in range(dim)]
                + ["Gradient" + str(i) for i in range(dim)]
                + ["Data_Type", "Simulation",
                   "Converged", "Norm_Term",
                   "Plus_Miss_Prop", "Minus_Miss_Prop",
                   "Optim_Iter", "Estimator"])

    optim_data_frame = pa.DataFrame(optim_list,
                                    columns=colnames)
    return optim_data_frame


# %%
# Simulation generation framework
def create_data_gen_np(
        z_plus_gen, z_minus_gen, nsiml, estimators, z_plus_gen_large, z_minus_gen_large,
        miss_func_plus=None, miss_func_minus=None, f=lambda x: x, impute=None,
        partial_miss=False, dat_vals_fun=get_dat_vals_impute,
        alpha=0.05, delta=0.05, **kwargs):
    """Performs simulated data generation and specified DRE approaches then performs NP classification using this information.

    Args:
        z_plus_gen (List): List of data generating functions to loop through
        z_minus_gen (List): List of negative class data generating functions to loop through
        nsiml (int): Number of simulations per scenario
        estimators (List): List of DRE estimators as functions (from estimators.torch)
        z_plus_gen_large (List): List of positive data generating for generating large data for power calculations
        z_minus_gen_large (List): List of negative data generating for generating large data for power calculations
        miss_func_plus (List, optional): List of positive class missing functions. Defaults to None.
        miss_func_minus (List, optional): List of negative class missing functions. Defaults to None.
        f (Function, optional): Function to apply in KLIEP standard form. Defaults to lambdax:x.
        impute (bool, optional): Whether imputed version of data should be produced. Defaults to True.
        partial_miss (bool, optional): Whether data is partially or fully missing
        dat_vals_fun (Function, optional): Function which summarises data for use with esimtators. Defaults to get_dat_vals_impute.
        alpha (float, optional): Target alpha in np classification. Defaults to 0.05.
        delta (float, optional): Target delta in np classification. Defaults to 0.05.

    Returns:
        Pandas.dataframe: A dataframe of the estimated parameters from each run and associated convergence diagnostics
    """

    # Get number of outer iterations
    nouter = len(z_plus_gen)
    out_data = {
        key: {key_1: [[] for iter in range(nouter)]
              for key_1 in ["poweralpha", "r_par", "classif"]}
        for key in estimators}
    # For each different scenario
    for i in range(nouter):
        if miss_func_plus is not None:
            varphi_plus = miss_func_plus[i]
        else:
            varphi_plus = None

        if miss_func_minus is not None:
            varphi_minus = miss_func_minus[i]
        else:
            varphi_minus = None

        # For each simulation
        for j in range(nsiml):
            # Create "true" data
            z_plus = z_plus_gen[i]()
            z_minus = z_minus_gen[i]()
            n_plus = z_plus.shape[0]
            n_minus = z_minus.shape[0]

            x_minus = z_minus.detach()
            x_plus = z_plus.detach()

            if partial_miss:
                if varphi_plus is not None:
                    x_plus = corrupt_func(x_plus, varphi_plus)
                if varphi_minus is not None:
                    x_minus = corrupt_func(x_minus, varphi_minus)

            else:
                # If required create corrupted + data
                if varphi_plus is not None:
                    # Create corrupted data
                    u_plus = unif.sample((n_plus,))
                    x_plus[
                        u_plus < varphi_plus(z_plus), :
                    ] = torch.nan

                # If required create corrupted - data
                if varphi_minus is not None:
                    # Create corrupted data
                    u_minus = unif.sample((n_minus))
                    x_minus[
                        u_minus < varphi_minus(z_minus), :
                    ] = torch.nan

            # remove perfect data
            del z_plus, z_minus

            # Compute data summaries
            dat = dat_vals_fun(
                x_plus=x_plus, x_minus=x_minus, f=f,
                varphi_plus=varphi_plus,
                varphi_minus=varphi_minus, impute=impute, **kwargs
            )

            # Perform DRE For each method
            est_dict = {key: func(dat, norm_fl=True, f=f, **kwargs)
                        for key, func in estimators.items()}

            # Create new data for simulation
            z_minus_classif = z_minus_gen[i]()
            # Create classifier for each method
            classif_dict = {
                key: cutoff_bin(est_dict[key]["r"], alpha=alpha, delta=delta,
                                newdata=z_minus_classif)
                for key in est_dict}
            z_plus_large = z_plus_gen_large[i]()
            z_minus_large = z_minus_gen_large[i]()

            result_dict = {
                key: power_alpha_calc(
                    classif[1], z_minus_large, z_plus_large)
                for key, classif in classif_dict.items()}

            for key in estimators:
                out_data[key]["poweralpha"][i].append(result_dict[key])
                out_data[key]["r_par"][i].append(est_dict[key]["par"])
                out_data[key]["classif"][i].append(classif_dict[key][0])

    return out_data
