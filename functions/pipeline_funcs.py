# %%
from sklearn.linear_model import LogisticRegression
import torch
import numpy as np
from torch import distributions
import sys
sys.path.append("..")
from functions.objective_funcs_torch import get_dat_vals_multidim  # noqa:E402
from functions.estimators_torch import kliep_multi_dim_sep_wrap  # noqa:E402
from functions.np_classifier_torch import cutoff_bin, power_alpha_calc  # noqa:E402
unif = distributions.Uniform(0, 1)
# %%


def get_ci(vec, verbose=True):
    """Get mean and CI for mean from vector

    Args:
        vec (Tensor): The vector of values you want the C.I. for the mean from
        verbose (bool, optional): Whether or not to print the CI. Defaults to True.
    """
    n = vec.shape[0]
    mean = torch.mean(vec, 0)
    se = torch.std(vec, 0)/(n**0.5)
    ci_up = mean+1.96*se
    ci_low = mean-1.96*se
    if verbose:
        print(f"Our Estimated Expected Power is: {mean:4.3f}")
        print(f"With ci({ci_low:4.3f}, {ci_up:4.3f})")
    return([mean, ci_low, ci_up])


def progress(percent=0, width=40):
    left = width * percent // 100
    right = width - left

    tags = "#" * left
    spaces = " " * right
    percents = f"{percent:.0f}%"

    print("\r[", tags, spaces, "]", percents, sep="", end="", flush=True)


def corrupt_func(data, missing_funcs, seed=None):
    """Corrupts data along each feature according to pre-specified missing functions

    Args:
        data (Tensor): The data to corrupt
        missing_funcs (List): A list of missing functions for each feature.
        If no missingness should be applied, put None.

    Returns:
        Tensor: Corrupted Data
    """
    if seed is not None:
        # print(f"Missing seed is {seed}")
        torch.manual_seed(seed)
    # Get uniform samples to select missing points
    us = unif.sample(data.shape)
    # Set up missing probabilities for each point
    misses = torch.zeros(data.shape)
    for j, miss_func in enumerate(missing_funcs):
        if miss_func is not None:
            # If missing function for column available
            #  use it to claculate missing prob
            misses[:, j] = miss_func(data[:, j])
    # Replace all points which should be missing with NaN
    nan = torch.nan
    out_data = torch.where(
        us < misses, nan, data.double()).float()

    return out_data


def miss_func_creator(theta):
    """Create logistic regression functions from parameters

    Args:
        theta (Tensor): A 2-element tensor containing the intercept and slope term.
    """
    def miss_func(x):
        return 1/(1+torch.exp((-theta[0]-theta[1]*x)))
    return miss_func


def create_standard_miss_func(mean, sd, sign, shift=0.):
    def miss_func(x):
        return(1/(1+torch.exp(sign*((x-mean)/sd)-shift)))
    return miss_func


def func_adj(m, std, f):
    """Change missing functions to take in normalised values

    Args:
        m (float): mean of original data
        std (float): The stnadard deviationn of original data
        f (function): The function to normalise

    Returns:
        function: The equivalent function to f for taking in normalised data.
    """
    if f is None:
        return None
    else:
        def miss_func(x):
            return f(x*std+m)
        return miss_func


def learn_missing_func(corrupted_data, clean_data, nlearn=50, seed=None):
    """A function to learn the missingness structure of the data
    by revealing the value of a few key points

    Args:
        corrupted_data (Tensor): Data with corruptions
        clean_data (Tensor): Fully observed data (should be identical to non-mising parts of corrupted data)
        nlearn (int, optional): The number of missing points to learn. Defaults to 50.

    Returns:
        List: List of estimated missing functions.
    """
    y = torch.isnan(corrupted_data).bool()
    est_miss_params = []
    # Set seed if given
    if seed is not None:
        torch.manual_seed(seed)
    for j in range(corrupted_data.shape[1]):
        # Get missing indices to learn
        which_miss = torch.nonzero(y[:, j]).reshape(-1)
        nmiss = which_miss.shape[0]
        x_adj = (corrupted_data[:, j]).clone().detach()
        # Learn those 50 values
        to_learn = which_miss[
            torch.multinomial(torch.zeros(which_miss.shape[0])+1, nlearn)]
        x_adj[to_learn] = clean_data[to_learn, j]
        no_miss2 = ~torch.isnan(x_adj)

        # ####Check if data completely separable####
        # non-missing data
        null_dat = x_adj[~y[:, j]]
        # learned missing data
        alt_dat = x_adj[to_learn]
        # Check if separated
        if torch.max(null_dat) < torch.min(alt_dat):
            midpoint = (torch.max(null_dat) + torch.min(alt_dat))/2
            params = torch.tensor([-100.*midpoint, 100.])
        elif torch.max(alt_dat) < torch.min(null_dat):
            midpoint = (torch.max(null_dat) + torch.min(alt_dat))/2
            params = torch.tensor([100.*midpoint, -100.])
        else:
            # Set up logistic regression
            model = LogisticRegression(
                solver='lbfgs', penalty='none')
            # Fit logistic regression
            model.fit((x_adj[no_miss2]).numpy().reshape(-1, 1),
                      (y[no_miss2, j]).numpy())
            # Adjust intercept
            a_0 = model.intercept_-np.log(nlearn/nmiss)
            # Bind parameters
            params = torch.tensor(
                [a_0[0], model.coef_[0, 0]])

        est_miss_params.append(params)
    # Create estimated missingness functions
    est_miss_funcs = [miss_func_creator(miss_param)
                      for miss_param in est_miss_params]
    return est_miss_funcs


def missing_pipeline(data_dict, missing_funcs, dr_proc=kliep_multi_dim_sep_wrap,
                     dat_val_fun=get_dat_vals_multidim,
                     norm=True, est_miss=True, lr=1, alpha=0.1, f=lambda x: x,
                     delta=0.1, nlearn=50, miss_seed=None, learn_seed=None, **kwargs):
    """Perform the entired DRE pipeline: corruption, normalisation, learn missingness, DRE, NP classificaiton

    Args:
        data_dict (dictionary): A dictionary containing all the partitions of the data. These are:
        alt_tr,null_tr,alt_test, null_cal.
        missing_funcs (list): list of missingness functions for data. If None no corruption performed.
        norm (bool, optional): Should normalisation be done. Defaults to True.
        est_miss (bool, optional): Should missing functions be estimated. Defaults to True.
        lr (list, optional): learning rate for DRE. Defaults to None.
        alpha (float/list, optional): List of alphas for NP classification. Defaults to 0.1.
        delta (float/list, optional): List of deltas for NP classification. Defaults to 0.1.
        nlearn (int, optional): Number of points used to learn missingness funcs. Defaults to 50.

    Returns:
        dictionary: contains power and dr fit
    """
    out_dict = {}
    model_dat_dict = data_dict.copy()

    # Covert alpha
    if type(alpha) == float:
        alpha = [alpha]
    if type(delta) == float:
        delta = [delta]

    # ### Optionally perform corruption ###
    if missing_funcs is not None:
        model_dat_dict["alt_tr"] = corrupt_func(
            model_dat_dict["alt_tr"], missing_funcs, seed=miss_seed)

    # ### Optionally Perform normalisation ###
    if norm:
        # Re-assign full_tr
        model_dat_dict["full_tr"] = torch.cat(
            (model_dat_dict["null_tr"], model_dat_dict["alt_tr"]),
            dim=0)
        # Get mean and sd
        miss_std = torch.tensor(
            np.nanstd(model_dat_dict["full_tr"], axis=0))
        miss_mean = torch.nanmean(model_dat_dict["full_tr"], dim=0)
        # Normalise data
        model_dat_dict = {key: (value-miss_mean)/miss_std for
                          key, value in model_dat_dict.items()}
        # Normalise
        if (not est_miss) & (missing_funcs is not None):
            missing_funcs = [
                func_adj(m, std, f) for f, m, std in
                zip(missing_funcs, miss_mean, miss_std)]
        # If estimated, normalise non-missing data as wekk
        else:
            data_dict = {key: (value-miss_mean)/miss_std for
                         key, value in data_dict.items()}
        out_dict["norm"] = {"mean": miss_mean, "std": miss_std}
    # ### Optionally learn missingness functions ###
    if est_miss:
        miss_funcs_to_use = learn_missing_func(
            corrupted_data=model_dat_dict["alt_tr"], clean_data=data_dict["alt_tr"],
            nlearn=nlearn, seed=learn_seed)
    else:
        miss_funcs_to_use = missing_funcs
    out_dict["est_miss_funcs"] = miss_funcs_to_use
    # ### Perform DRE ###
    # Do initial data summaries
    gen_dat_miss = dat_val_fun(
        x_plus=model_dat_dict["alt_tr"], x_minus=model_dat_dict["null_tr"],
        varphi_plus=miss_funcs_to_use, varphi_minus=None, f=f, **kwargs
    )
    # Do actual DRE
    miss_result = dr_proc(
        gen_dat_miss, norm_fl=True, lr=lr, f=f, **kwargs)
    # Extract estimated r
    miss_est_dr = miss_result["r"]
    out_dict["dr"] = miss_result

    out_dict["power_res"] = []
    # ###NP Classification### #
    # For each alpha-delta combination construct classifier.
    for j in range(len(alpha)):
        # Get NP classifier
        threshold, classif = cutoff_bin(
            class_func=miss_est_dr, alpha=alpha[j], delta=delta[j],
            newdata=model_dat_dict["null_cal"])

        # Asses classifier
        power_res = power_alpha_calc(
            classif=classif, data_0=model_dat_dict["null_cal"],
            data_1=model_dat_dict["alt_test"])
        out_dict["power_res"].append(power_res)

    out_dict["alphas"] = alpha
    out_dict["deltas"] = delta
    out_dict["prop_miss"] = (
        torch.sum(torch.isnan(model_dat_dict["alt_tr"]))
        / torch.numel(model_dat_dict["alt_tr"]))

    return out_dict


def full_pipeline(null_df, alt_df, missing_funcs, n_altte, n_nulltr,
                  dr_proc=kliep_multi_dim_sep_wrap,
                  dat_val_fun=get_dat_vals_multidim,
                  norm=True, est_miss=True, lr=1, alpha=0.1, f=lambda x: x,
                  delta=0.1, nlearn=50, miss_seed=None, learn_seed=None,
                  split_seed=None, **kwargs):
    split_dict = {}
    # Split these into test, train, and calibration
    split_dict["null_tr"] = null_df.sample(n_nulltr, random_state=split_seed)
    split_dict["null_cal"] = null_df.drop(
        split_dict["null_tr"].index)

    split_dict["alt_test"] = alt_df.sample(n_altte, random_state=split_seed)
    split_dict["alt_tr"] = alt_df.drop(
        split_dict["alt_test"].index)

    # Get tensor versions
    # null_tens = torch.tensor(null_df[]
    split_tens = {}
    for key, value in split_dict.items():
        split_tens[key] = torch.tensor(
            (value).to_numpy().astype(np.float32))
    d = split_tens["null_tr"].shape[1]
    # Impute each row of the tensor
    nanmeans = {}
    # Do original imputation to have only our own missingness
    for key, value in split_tens.items():
        nanmeans[key] = torch.nanmean(value, dim=0)
        for i in range(d):
            split_tens[key][torch.isnan(value[:, i]), i] = nanmeans[key][i]

    out = missing_pipeline(
        split_tens, missing_funcs, dr_proc=dr_proc, dat_val_fun=dat_val_fun,
        est_miss=est_miss, lr=lr, alpha=alpha, delta=delta, f=f, nlearn=nlearn,
        norm=norm, miss_seed=miss_seed, learn_seed=learn_seed, **kwargs)
    return out
