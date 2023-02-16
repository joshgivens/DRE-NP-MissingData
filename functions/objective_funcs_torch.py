# %%
import torch
import numpy
# explicitly require this experimental feature
from sklearn.experimental import enable_iterative_imputer  # noqa
# now you can import normally from impute
from sklearn.impute import IterativeImputer
# %%


def get_dat_vals_impute_sing(x, f=lambda x: x, varphi=None, impute=None,
                             add_args=None, **kwargs):
    """Get data summaries for quicker implementation of KLIEP etc.

    Args:
        x (Tensor): Corrupted data
        f (function, optional): f for use in log-linear parametric form. Defaults to lambdax:x.
        varphi(function, optional): Missingness function. Defaults to None.
        impute (bool, optional): Whether to create imputed data as well. Defaults to False.

    Returns:
        Dictionary: A dictionary of data summaries to use in M-KLIEP etc
    """
    if add_args is None:
        add_args = {}
    # Gives KL divergence for density ratio with given theta, data and function
    n = x.shape[0]

    # Only get missing values for plus if required
    if (varphi is not None):
        # Count number of missing values in each group
        isnan_x = torch.isnan(x[:, 0])
        nmiss = int(sum(isnan_x))

        # Update to remove nans
        x_filt = x[~isnan_x, :]
        varphi_val = varphi(x_filt)
        weight = varphi_val/(1-varphi_val)

    else:
        x_filt = x
        weight = None
        varphi_val = 0
        nmiss = 0

    # Check if any imputation required
    if impute is not None:
        # Check to see if imputation required for +
        if nmiss > 0:
            # Get probs from weights
            sum_weight = torch.sum(weight)
            if sum_weight == 0:
                probs = None
                print("No relevant points to impute from for +")
            else:
                probs = weight/sum_weight

            # Get random weighted indexes
            plus_ind = torch.multinomial(
                probs, num_samples=nmiss, replacement=True)

            # Use indexes to imput missing values
            x_impute = torch.cat((x_filt, x_filt[plus_ind, :]), axis=0)

        # Else plus requires no impute then set to original
        else:
            x_impute = x

    # Else if no imputation required
    else:
        x_impute = None

    # Check if we want to transform x
    if f is not None:
        # Get functions of positive and negative data as matrix
        # Row i is f(x_i)
        f_val = f(x_filt, **add_args)
        # If imputation asked for and some missing do
        if impute:
            if nmiss > 0:
                f_impute = f(x_impute, **add_args)
            else:
                f_impute = f_val

        # If no imputation whatsoever
        else:
            f_impute = None

    # Else if f not required
    else:
        f_val = None
        f_impute = None

    return {
        "f_val": f_val, "x": x_filt, "x_impute": x_impute,
        "nmiss": nmiss, "weight": weight, "n": n,
        "f_impute": f_impute, "varphi_val": varphi_val}


def get_dat_vals_impute(x_plus, x_minus, f=lambda x: x, varphi_plus=None,
                        varphi_minus=None, impute=False, add_args=None, **kwargs):
    """Get data summaries for quicker implementation of KLIEP etc.

    Args:
        x_plus (Tensor): Class 1 data
        x_minus (Tensor): Class 0 data
        f (function, optional): f for use in log-linear parametric form. Defaults to lambdax:x.
        varphi_plus (function, optional): class 1 missingness function. Defaults to None.
        varphi_minus (function, optional): class 0 missingness function. Defaults to None.
        impute (bool, optional): Whether to create imputed data as well. Defaults to False.

    Returns:
        Dictionary: A dictionary of data summaries to use in M-KLIEP etc
    """
    plus_vals = get_dat_vals_impute_sing(
        x_plus, f, varphi_plus, impute, add_args, **kwargs)
    minus_vals = get_dat_vals_impute_sing(
        x_minus, f, varphi_minus, impute, add_args, **kwargs)

    plus_vals = {key+"_plus": value for key, value in plus_vals.items()}
    minus_vals = {key+"_minus": value for key, value in minus_vals.items()}
    return plus_vals | minus_vals


def neg_kliep_miss_f(theta, f_val_plus, f_val_minus, n_plus, n_minus,
                     varphi_val_plus=None, varphi_val_minus=None, reg=0.):
    """Negative M-KLIEP objective

    Args:
        theta (Tensor): Parameter
        f_val_plus (Tensor): f evaulated on class 1 data
        f_val_minus (Tensor): f evaluated on class 0 data
        n_plus (int): number of class 1 samples
        n_minus (int): number of class 0 samples
        varphi_val_plus (Tensor, optional): Class 1 missingness function evaulated on non-missing class 1 data. Defaults to None.
        varphi_val_minus (Tensor, optional): Class 0 missingness function evaulated on non-missing class 0 data. Defaults to None.
        reg (float, optional): L2 regularisation term. Defaults to 0..

    Returns:
        float: objective evaluated at theta with given data
    """
    if type(theta) != torch.nn.Parameter:
        if type(theta) != torch.Tensor:
            theta = torch.tensor(theta).float()
        else:
            theta = theta.float()

    plus_term = (f_val_plus@theta.reshape(-1))
    minus_term = torch.exp(f_val_minus@theta.reshape(-1))

    if (varphi_val_plus is not None) & (type(varphi_val_plus) != int):
        weight_plus = (1-varphi_val_plus).reshape(-1)
    else:
        weight_plus = 1
    if (varphi_val_minus is not None) & (type(varphi_val_minus) != int):
        weight_minus = (1-varphi_val_minus).reshape(-1)
    else:
        weight_minus = 1
    out_plus = torch.sum(plus_term/weight_plus)/n_plus
    out_minus = torch.log(torch.sum(minus_term/weight_minus)/n_minus)

    return -(out_plus-out_minus)+reg*torch.linalg.vector_norm(theta)


def nkliep_theta_approx_fast(theta, f_val_minus, n_minus, varphi_val_minus=None):
    """Get M-KLIEP normalisation term

    Args:
        theta (Tensor): Parameter
        f_val_minus (Tensor): F evaulated at Non-missing class 1 data
        n_minus (int): Number of class 1 observations
        varphi_val_plus (Tensor, optional): Class 1 missingness function evaulated on non-missing class 1 data. Defaults to None.

    Returns:
        float: Normalisation term.
    """
    if type(theta) != torch.Tensor:
        theta = torch.tensor(theta).float()
    else:
        theta = theta.float()
    minus_term = torch.exp(f_val_minus@theta.reshape(-1))
    if (varphi_val_minus is not None) & (type(varphi_val_minus) != int):
        weight_minus = (1-varphi_val_minus).reshape(-1)
    else:
        weight_minus = 1
    out_minus = torch.sum(minus_term/weight_minus)/n_minus
    return out_minus


def get_unnorm_weights(x, varphi):
    if varphi is None:
        return None
    else:
        return varphi(x)


def get_weights(x, varphi):
    if varphi is None:
        return None
    else:
        return varphi(x)/(1-varphi(x))


def get_dat_vals_multidim_sing(
        x, varphi, f=lambda x: x, impute=None, mice_args=None, **kwargs):
    """Get summary data values for multi-dimensional case

    Args:
        x (Tensor): Corrupted data
        varphi (List): List of missing function
        f (function, optional): Your f function you are using for each dimension.
            Defaults to lambdax:x.
        impute (string, optional): Imputation method to use. Can be one of: "MICE", "Weighted", or "Uniform".
            Defatults to None.
        mice_args (dict, optional): Additional arguments to IterativeImputer. Defaults to None.

    Returns:
        Dictionairy: A dictionary of useful summaries. These are:
        "x_filt", "f_val_plus/minus", "isnan_x_plus/minus",
        "nmiss_plus/minus", "weight_plus/minus"
    """
    if mice_args is None:
        mice_args = {}
    # get dimension
    d = x.shape[1]
    if varphi is not None:
        isnan_x = torch.isnan(x)
        nmiss = torch.sum(isnan_x, axis=0)
        # Get each filtered row as a 2D Tensor with width 1.
        x_filt = [x[~isnan_x[:, j], j:j+1] for j in range(d)]
        f_val = [f(col) for col in x_filt]
        varphi_val = [
            get_unnorm_weights(col, varphi) for
            col, varphi in zip(x_filt, varphi)
        ]
        weight = [
            get_weights(col, varphi) for
            col, varphi in zip(x_filt, varphi)]

    else:
        isnan_x = torch.zeros((x.shape[0], d))
        nmiss = torch.zeros(d)
        x_filt = [x[:, j:j+1] for j in range(d)]
        f_val = [f(col) for col in x_filt]
        weight = [None]*d
        varphi_val = [None]*d

    # Do optional imputation
    if impute is not None:
        # If MICE do MICE imputation
        if impute.upper() == "MICE":
            imp = IterativeImputer(**mice_args)
            x_imp = torch.tensor(imp.fit_transform(x)).float()
        # Otherwise marginal imputation
        else:
            # Copy data to then imput missing values
            x_imp = x.detach().clone()

            # Accross each dimension
            for j in range(d):
                if nmiss[j] > 0:
                    # Get missing value locations
                    miss_ind = torch.nonzero(isnan_x[:, j])[:, 0]
                    # If impute is Weighted give weightings
                    if impute.upper() == "WEIGHTED":
                        probs = weight[j][:, 0]
                    # Else if uniform give no weightings (weights=1)
                    elif impute.upper() == "UNIFORM":
                        probs = torch.zeros((x_filt[j].shape[0]))+1
                    else:
                        raise Exception(
                            "Invalid impute option. Choices are: \"MICE\", \"Weighted\", \"Uniform\"")

                    # Randomly choose indices of filtered non-missing data
                    plus_ind = torch.multinomial(
                        probs, num_samples=nmiss[j], replacement=True)

                    # Replace missing data with randomly chosen indices
                    x_imp[miss_ind, j] = x_filt[j][plus_ind, 0]

        # Transform imputed data via f
        # f designed to be used on individual columns so concatenate trnasformation on each column.
        f_imp = [f(x_imp[:, j:j+1]) for j in range(d)]

    else:
        f_imp = None

    return {"x_filt": x_filt, "f_val": f_val, "isnan_x": isnan_x,
            "nmiss": nmiss, "weight": weight,
            "varphi_val": varphi_val, "f_imp": f_imp, }


def get_dat_vals_multidim(x_plus, x_minus, varphi_plus, varphi_minus, f=lambda x: x,
                          impute=None, mice_args=None, **kwargs):
    """Get summary data values for multi-dimensional case

    Args:
        x (Tensor): Positive class data
        x_minus (Tensor): Negative class data
        varphi_plus (List): List of positive missing function
        varphi_minus (List): List of negative missing function
        f (function, optional): Your f function you are using for each dimension.
        Defaults to lambdax:x.

    Returns:
        Dictionairy: A dictionary of useful summaries. These are:
        "x_plus/minus_filt", "f_val_plus/minus", "isnan_x_plus/minus",
        "nmiss_plus/minus", "weight_plus/minus, "varphi_val_plus/minus"
    """
    plus_vals = get_dat_vals_multidim_sing(
        x_plus, varphi_plus, f, impute, mice_args, **kwargs)
    minus_vals = get_dat_vals_multidim_sing(
        x_minus, varphi_minus, f, impute, mice_args, **kwargs)
    plus_vals = {key+"_plus": value for key, value in plus_vals.items()}
    minus_vals = {key+"_minus": value for key, value in minus_vals.items()}
    return plus_vals | minus_vals


def gr_kliep_miss_f(theta, f_val_plus, f_val_minus, n_plus, n_minus,
                    varphi_val_plus=None, varphi_val_minus=None, reg=0.,
                    **kwargs):
    """Gives gradient of M-KLIEP at given parameter

    Args:
        theta (Tensor): Parameter
        f_val_plus (Tensor): f evaulated on class 1 data
        f_val_minus (Tensor): f evaluated on class 0 data
        n_plus (int): number of class 1 samples
        n_minus (int): number of class 0 samples
        varphi_val_plus (Tensor, optional): Class 1 missingness function evaulated on non-missing class 1 data. Defaults to None.
        varphi_val_minus (Tensor, optional): Class 0 missingness function evaulated on non-missing class 0 data. Defaults to None.
        reg (float, optional): L2 regularisation term. Defaults to 0..

    Returns:
        float: gradient of M_KLIEP objective at theta with given data.
    """

    if type(theta) == numpy.ndarray:
        theta = torch.tensor(theta).float()
    theta = theta.clone().detach().requires_grad_(True)
    out = neg_kliep_miss_f(theta, f_val_plus, f_val_minus, n_plus, n_minus,
                           varphi_val_plus, varphi_val_minus, reg=reg)
    out.backward()
    return theta.grad


def adjusted_logreg(theta, x, y):
    """Performs logisitc regression with some inputs uniformly missing in each class.

    Args:
        theta (Tensor): 1-dim choice of parameters parameter 1 is log-odds.
         Other parameters are density ratio for density ratio.
        x (Tensor): Each row is an observsation from x. NAN represents missing input value
        y (Tensor): 1-dim. Labels given as 0 or 1.

    Returns:
        Float: The log-likelihood of the observations in our current model.
    """
    # Recode y from {0,1} to {-1,1}
    signs = 1-2*y
    # Filter
    x_nan = torch.isnan(x)
    signs_nonan = signs[~x_nan]
    x_nonan = x[~x_nan]
    signs_nan = signs[x_nan]
    no_miss_lik = torch.sum(torch.log1p(
        torch.exp(signs_nonan*(theta[0]+theta[1]+theta[2]*x_nonan)))
    )
    miss_lik = torch.sum(torch.log1p(
        torch.exp(signs_nan*theta[0]))
    )
    return (no_miss_lik+miss_lik)/y.shape[0]


def gr_adj_logreg(theta, x, y):
    theta = (torch.tensor(theta)).clone().detach().requires_grad_(True)
    out = adjusted_logreg(theta, x, y)
    out.backward()
    return theta.grad
