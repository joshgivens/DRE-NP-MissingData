# %%
import torch
import numpy as np
from math import e
from scipy.stats import binom

# %%


def cutoff_VP(class_func, alpha, delta, newdata):
    """Gets Classifier threshold via procedure in Tong2013

    Args:
        class_func (Function): Score function to be used for classification. Normally DR estimate.
        alpha (float): Target Type I error rate
        delta (float): Probability for target Type I error rate.
        newdata (Tensor): New Class 0 data to fit threshold with

    Returns:
        (float, Function): Returns the classification threshold and associated classifier.
    """
    n = newdata.shape[0]
    dn = 2*np.sqrt(2*(np.log(2*e*n)+np.log(2/delta))/n)
    Z = class_func(newdata)
    Z.sort()

    t = int(np.floor((alpha-dn)*n))
    if t == 0:
        C = Z[-1]+1e-5
    else:
        C = Z[n-t]

    # Create classifier
    def classif(x):
        (class_func(x) >= C).float()

    return (C, classif)


def cutoff_bin(class_func, alpha, delta, newdata):
    """Gets Classifier threshold via procedure in Tong2018

    Args:
        class_func (Function): Score function to be used for classification. Normally DR estimate.
        alpha (float): Target Type I error rate
        delta (float): Probability for target Type I error rate.
        newdata (Tensor): New Class 0 data to fit threshold with

    Returns:
        (float, Function): Returns the classification threshold and associated classifier.
    """
    n = newdata.shape[0]
    k = int(binom.ppf(1-delta, n, 1-alpha)) + 1
    if (k > n):
        C = 1e7
        print("No such classifier exists, set threshold arbitrarily high")
    else:
        Z = class_func(newdata)
        C = torch.kthvalue(Z, k)[0]

    def classif(x):
        return (class_func(x) >= C).float()

    return (C, classif)


def cutoff_hoeffding(class_func, alpha, delta, newdata,
                     miss_func, miss_max):
    """Gives Classification Boundary for NP classification
    with missing data use Hoeffding Bound

    Args:
        class_func (function): The estimate density ratio function
        alpha (float): Desired Type I error
        delta (float): Disred probability guarantee for Type I error
        newdata (Tensor): New null data for threshold selection
        miss_func (function): Missingness function for null class
        miss_max (float): Maximum value missingness function can take.

    Returns:
        "C" (float): Numerical threshold for the density ratio
        "classif" (function): Classification Function.
    """
    n = newdata.shape[0]
    w_threshold = (1-miss_max)*torch.sqrt(
        -torch.log(delta)/2*n)

    Z = class_func(newdata)
    w = (torch.isnan(newdata[:, 0]).float()
         / (1-miss_func(newdata).reshape(-1)))
    Z_sorted, indices = Z.sort()
    w = w[indices]
    w_cumsum = torch.cumsum(w)
    i_star = torch.min(torch.nonzero(
        torch.where(w_cumsum <= alpha-w_threshold)
    ))
    C = Z_sorted[i_star]

    def classif(x):
        return (class_func(x) >= C).float()

    return (C, classif)


def cutoff_bernstein(class_func, alpha, delta, newdata,
                     miss_func, miss_max):
    """Gives Classification Boundary for NP classification
    with missing data using  Bound

    Args:
        class_func (function): The estimate density ratio function
        alpha (float): Desired Type I error
        delta (float): Disred probability guarantee for Type I error
        newdata (Tensor): New null data for threshold selection
        miss_func (function): Missingness function for null class
        miss_max (float): Maximum value missingness function can take.

    Returns:
        "C" (float): Numerical threshold for the density ratio
        "classif" (function): Classification Function.
    """
    n = newdata.shape[0]
    w_threshold = torch.sqrt((2-8*torch.log(delta))
                             / n*(1-miss_max))

    Z = class_func(newdata)
    w = (torch.isnan(newdata[:, 0]).float()
         / (1-miss_func(newdata).reshape(-1)))
    Z_sorted, indices = Z.sort()
    w = w[indices]
    w_cumsum = torch.cumsum(w)
    i_star = torch.min(torch.nonzero(
        torch.where(w_cumsum <= alpha-w_threshold)
    ))
    C = Z_sorted[i_star]

    def classif(x):
        return (class_func(x) >= C).float()

    return (C, classif)


def power_alpha_calc(classif, data_0, data_1):
    """Gives power and Type I error of classifier on new data

    Args:
        classif (Function): The classifier you want to asses
        data_0 (Tensor): New class 0 data
        data_1 (Tensor): New class 1 data

    Returns:
        [float, float]: the power and Type I error of your classifier on the new data.
    """
    class_1 = classif(data_1)
    class_0 = classif(data_0)
    power = torch.mean(class_1)
    alpha = torch.mean(class_0)
    return [power, alpha]


def classify_from_df(df, create_ratio_func, alpha, delta,
                     z_minus_gen, z_plus_gen_large, z_minus_gen_large):
    """Create a classifier and tests the power from parameters in dataframe

    Args:
        df (Pandas.dataframe): Dataframe with parameters for various classifiers. Each row is a new classifier
        create_ratio_func (Function): Function which takes in parameters and produces score function.
        alpha (float): Target alpha
        delta (float): Probability of not reaching target alpha
        z_minus_gen (function): A function to produce data to create classifier from
        z_plus_gen_large (Function): A function to produce a large amount of class 1 data
        z_minus_gen_large (Function): A function to produce a large amount of class 0 data

    Returns:
        Pandas.dataframe: The original dataframe with power and alpha columns appended.
    """
    nsamp = df.shape[0]
    power_list = []
    alpha_list = []
    C_list = []
    dat_temp = -1

    data_type = df["Data_Type"]
    Param_cols = [col for col in df.columns if 'Param' in col]
    for i in range(nsamp):
        new_sim = (dat_temp != df.loc[i, "Simulation"])

        theta = torch.tensor(df.loc[i, Param_cols]).float()

        temp_class_func = create_ratio_func(theta)
        data_index = data_type[i]
        # Generate classifying data
        if new_sim:
            # Generate classifying data
            x_0_new = z_minus_gen[data_index]()

        temp_classif = cutoff_bin(temp_class_func, alpha, delta, x_0_new)

        if new_sim:
            # Now estimate power and alpha
            power_test_dat = z_plus_gen_large[data_index]()
            alpha_test_dat = z_minus_gen_large[data_index]()

        power_emp, alpha_emp = power_alpha_calc(
            temp_classif[1], alpha_test_dat, power_test_dat
        )

        # now append to data frame
        power_list.append(power_emp)
        alpha_list.append(alpha_emp)
        C_list.append(temp_classif[0])

        dat_temp = df.loc[i, "Simulation"]

    df["power"] = np.array(power_list)
    df["alpha"] = np.array(alpha_list)
    df["cutoff"] = np.array(C_list)
    return df
# %%
