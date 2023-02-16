# %%
import torch
from torch import distributions as dists
import numpy as np
# from scipy import stats
import sys
import matplotlib.pyplot as plt
import pickle
sys.path.append("..")
from functions.data_sim_framework_torch import create_data_gen, create_data_gen_np  # noqa:E402
import functions.estimators_torch as est  # noqa:E402
from functions.objective_funcs_torch import get_dat_vals_multidim  # noqa:E402
from functions.np_classifier_torch import classify_from_df, cutoff_bin, power_alpha_calc  # noqa: E402

plt.rcParams["figure.facecolor"] = "White"
plt.rcParams["axes.facecolor"] = "White"
plt.rcParams["savefig.facecolor"] = "White"


def progress(percent=0, width=40):
    left = width * percent // 100
    right = width - left

    tags = "#" * left
    spaces = " " * right
    percents = f"{percent:.0f}%"

    print("\r[", tags, spaces, "]", percents, sep="", end="", flush=True)


def sampler_creator(n, dist):
    def sampler():
        return dist.sample((n, 1))
    return sampler


def mv_sampler_creator(n, dist):
    def sampler():
        return dist.sample((n,))
    return sampler


def mv_mix_sampler_creator(n, dist_1, dist_2, p=0.5):
    def sampler():
        u = dists.Binomial(n, p).sample((1,))[0]
        samp_1 = dist_1.sample((int(u),))
        samp_2 = dist_2.sample((int(n-u),))
        return torch.concat([samp_1, samp_2])
    return sampler


def dr_creator(plus_dist, minus_dist):
    def true_r(x):
        return plus_dist.log_prob(x)-minus_dist.log_prob(x)
    return true_r


lr = 0.7**(np.floor((np.arange(1000))/100)+1)
# %%
# ################## First example Multivariate Gaussian Example #########################
ns = np.arange(100, 1501, 100)
# Generate data generating procedures
plus_gen_list = [mv_sampler_creator(
    n, dists.MultivariateNormal(torch.zeros(5), torch.eye(5)))
    for n in ns]
minus_gen_list = [mv_sampler_creator(
    n, dists.MultivariateNormal(torch.zeros(5)+0.1, torch.eye(5)))
    for n in ns]
p_0 = 0.5


# Create missing function
def miss_func(x):
    return torch.where(torch.sum(x, 1) > 0.,
                       p_0, 0.)


miss_func_list = [miss_func]*ns.shape[0]

# %%
nsiml = 100
data = create_data_gen(
    plus_gen_list, minus_gen_list, nsiml=nsiml,
    estimators={"KLIEP Miss": est.kliep_miss_wrap,
                "KLIEP Naive": est.kliep_naive_wrap}, impute=False,
    miss_func_plus=miss_func_list, miss_func_minus=None, maxiter=100)

out_dict = {"Data": data, "Param": ns}

with open('../results/simulated_results/Vary_n_one_class_5dim_'+str(nsiml)
          + 'sim_comp_diff=0.1_torch.pkl', 'wb') as handle:
    pickle.dump(out_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

print("Multi-dimensional Asymmetric case completed")

# %%
# ################ Partial Missingness Naive Bayes Setting  ########################################
lr = 0.7**(np.floor((np.arange(1000))/100)+1)
ns = np.arange(100, 1501, 100)
# Generate data generating procedures
cov_mats = [torch.tensor([[1, i], [i, 1]]).float()
            for i in np.linspace(0, 0.9, 10)]

plus_dists = [dists.MultivariateNormal(
    torch.zeros(2), cov) for cov in cov_mats]
minus_dists = [dists.MultivariateNormal(
    torch.tensor([1., 2.]), cov) for cov in cov_mats]

plus_gen_list = [mv_sampler_creator(100, plus_dist)
                 for plus_dist in plus_dists]
minus_gen_list = [mv_sampler_creator(100, minus_dist)
                  for minus_dist in minus_dists]

plus_gen_list_large = [mv_sampler_creator(
    int(1e6), plus_dist) for plus_dist in plus_dists]
minus_gen_list_large = [mv_sampler_creator(
    int(1e6), minus_dist) for minus_dist in minus_dists]

p_0 = 0.8


# Create missing function
def miss_func(x):
    return torch.where(x > 0., p_0, 0.)


def miss_func2(x):
    return torch.where(x < 0., p_0, 0.)


miss_func_list = [[miss_func, miss_func2] for iter in range(ns.shape[0])]

# %%
nsiml = 100
data = create_data_gen_np(
    plus_gen_list, minus_gen_list, nsiml=nsiml, z_plus_gen_large=plus_gen_list_large,
    z_minus_gen_large=minus_gen_list_large,
    estimators={"KLIEP Miss": est.kliep_multi_dim_sep_wrap,
                "KLIEP Naive": est.kliep_multi_dim_naive_sep_wrap,
                "KLIEP Impute": est.kliep_multi_dim_imp_wrap,
                "KLIEP Impute Sep": est.kliep_multi_dim_sep_imp_wrap},
    miss_func_plus=miss_func_list, miss_func_minus=None, partial_miss=True,
    dat_vals_fun=get_dat_vals_multidim, maxiter=100, alpha=0.1, delta=0.1,
    impute="Uniform")

# %%

true_rs = [dr_creator(plus_dist, minus_dist)
           for plus_dist, minus_dist in zip(plus_dists, minus_dists)]

# Do the same for known r
true_r_data = {key: [[] for iter in range(len(plus_gen_list))]
               for key in ["poweralpha", "classif"]}

for j, temp_true_r in enumerate(true_rs):
    for i in range(nsiml):
        c, classif = cutoff_bin(
            temp_true_r, alpha=0.1, delta=0.1, newdata=minus_gen_list[j]())
        power = power_alpha_calc(classif, minus_gen_list_large[j](),
                                 plus_gen_list_large[j]())
        true_r_data["poweralpha"][j].append(power)
        true_r_data["classif"][j].append(c)

# %%


def temp_r(x):
    return plus_dists[0].log_prob(x)-minus_dists[0].log_prob(x)


print(temp_r(torch.tensor([1, 2])))
print(true_rs[0](torch.tensor([1, 2])))
print(true_rs[1](torch.tensor([1, 2])))

c, classif = cutoff_bin(
    temp_r, alpha=0.1, delta=0.1, newdata=minus_gen_list[0]())
power = power_alpha_calc(classif, minus_gen_list_large[0](),
                         plus_gen_list_large[0]())
power
# %%
data["True R"] = true_r_data


out_dict = {"Data": data}

with open('../results/simulated_results/vary_cor_'+str(nsiml)
          + 'sim_comp_diff=0.1_torch.pkl', 'wb') as handle:
    pickle.dump(out_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

print("Multi-dimensional Asymmetric case completed")

# %%
# ################ Incorrectly Specified 1st Setting ##################
# Create dists
# Create positive data generation function
# (generates 100 normal(0,1) )
ns = np.arange(100, 1501, 100)
z_plus_0 = dists.MultivariateNormal(
    torch.zeros(2), torch.eye(2))
z_plus_1 = dists.MultivariateNormal(
    torch.tensor([-1., 4.]), torch.eye(2))


z_minus_0 = dists.MultivariateNormal(
    torch.tensor([1., 0.]), torch.eye(2))
z_minus_1 = dists.MultivariateNormal(
    torch.tensor([0., 4.]), torch.eye(2))


plus_gen_list = [mv_mix_sampler_creator(
    n, z_plus_0, z_plus_1, 0.5) for n in ns]

minus_gen_list = [mv_mix_sampler_creator(
    n, z_minus_0, z_minus_1, 0.5) for n in ns]

# Create miss_plus and miss_minus list
p_0 = 0.9


def temp_miss_func(x):
    return torch.where(x[:, 1] < 2., 0., p_0)


miss_func_list = [temp_miss_func]*ns.shape[0]
# %%
nsiml = 10
lr = (0.7**(np.floor((np.arange(1000)+2)/100)))
data = create_data_gen(
    plus_gen_list, minus_gen_list, nsiml=nsiml,
    estimators={"KLIEP Miss": est.kliep_miss_wrap,
                "KLIEP Naive": est.kliep_naive_wrap}, impute=False,
    miss_func_plus=miss_func_list, miss_func_minus=None, maxiter=1000,
    lr=lr)

out_dict = {"Data": data, "Param": ns}

with open('../results/simulated_results/NP_mixed_classif_aistat_'+str(nsiml)
          + 'sim_.pkl', 'wb') as handle:
    pickle.dump(out_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

print("NP classification completed")

# %%
with open('../results/simulated_results/NP_mixed_classif_aistat_'+str(nsiml)
          + 'sim_.pkl', 'rb') as handle:
    Output = pickle.load(handle)
df = Output["Data"]


def create_class_func(theta):
    def class_func(x):
        return torch.exp(x@theta).reshape(-1)
    return class_func


z_plus_large = [mv_mix_sampler_creator(
    int(1e6), z_plus_0, z_plus_1, 0.5)]*ns.shape[0]
z_minus_large = [mv_mix_sampler_creator(
    int(1e6), z_minus_0, z_minus_1, 0.5)]*ns.shape[0]


def true_r(x):
    return ((0.5*torch.exp(z_plus_0.log_prob(x))+0.5*torch.exp(z_plus_1.log_prob(x)))
            / (0.5*torch.exp(z_minus_0.log_prob(x))+0.5*torch.exp(z_minus_1.log_prob(x))))


power_res = []
for j, dot in enumerate(ns):
    power_res_temp = []
    for i in range(100):
        threshold, classifier = cutoff_bin(
            true_r, alpha=0.1, delta=0.1, newdata=minus_gen_list[j]())

        power_res_temp.append(power_alpha_calc(
            classifier, data_0=z_minus_large[j](), data_1=z_plus_large[j]()))
    power_res.append(power_res_temp)

df = classify_from_df(df, create_class_func, 0.1, 0.1,
                      minus_gen_list, z_plus_large, z_minus_large)
out_dict = {"Data": df, "Param": ns, "True_r_res": power_res}

with open('../results/simulated_results/NP_mixed_classif_aistat_'+str(nsiml)
          + 'sim_.pkl', 'wb') as handle:
    pickle.dump(out_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

print("Simulation Finished")

# %%
# Perform NP classification with Optimal Classification Thresholds
with open('../results/simulated_results/NP_mixed_classif_aistat_'+str(nsiml)
          + 'sim_.pkl', 'rb') as handle:
    Output = pickle.load(handle)
df = Output["Data"]


def create_class_func(theta):
    def class_func(x):
        return torch.exp(x@theta).reshape(-1)
    return class_func


z_plus_large = [mv_mix_sampler_creator(
    int(1e6), z_plus_0, z_plus_1, 0.5)]*ns.shape[0]
z_minus_large = [mv_mix_sampler_creator(
    int(1e6), z_minus_0, z_minus_1, 0.5)]*ns.shape[0]


def true_r(x):
    return ((0.5*torch.exp(z_plus_0.log_prob(x))+0.5*torch.exp(z_plus_1.log_prob(x)))
            / (0.5*torch.exp(z_minus_0.log_prob(x))+0.5*torch.exp(z_minus_1.log_prob(x))))


power_res = []
for j, dot in enumerate(ns):
    power_res_temp = []
    for i in range(100):
        threshold, classifier = cutoff_bin(
            true_r, alpha=0.1, delta=0.1, newdata=z_minus_large[j]())

        power_res_temp.append(power_alpha_calc(
            classifier, data_0=z_minus_large[j](), data_1=z_plus_large[j]()))
    power_res.append(power_res_temp)

df = classify_from_df(df, create_class_func, 0.1, 0.1,
                      z_minus_large, z_plus_large, z_minus_large)
out_dict = {"Data": df, "Param": ns, "True_r_res": power_res}

with open('../results/simulated_results/NP_mixed_classif_aistat_'+str(nsiml)
          + 'sim_largeclassdat.pkl', 'wb') as handle:
    pickle.dump(out_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

print("Simulation Finished")

# %%
# ################ Incorrectly specified 2nd Setting ##################
lr = 0.7**(np.floor((np.arange(1000))/100)+1)
ns = np.arange(100, 1501, 100)
# Generate data generating procedures
cov_mat = torch.tensor([[2, 1], [1, 1]]).float()
diff = 1

plus_dist = dists.MultivariateNormal(torch.zeros(2), torch.eye(2))
minus_dist = dists.MultivariateNormal(torch.zeros(2)+diff, cov_mat)

plus_gen_list = [mv_sampler_creator(n, plus_dist)
                 for n in ns]
minus_gen_list = [mv_sampler_creator(n, minus_dist)
                  for n in ns]

plus_gen_list_large = [
    mv_sampler_creator(int(1e6), plus_dist) for n in ns]
minus_gen_list_large = [
    mv_sampler_creator(int(1e6), minus_dist) for n in ns]

p_0 = 0.8


# Create missing function
def miss_func(x):
    return torch.where(x[:, 1] > 0., p_0, 0.)


def true_r(x):
    return plus_dist.log_prob(x)-minus_dist.log_prob(x)


miss_func_list = [miss_func for iter in range(ns.shape[0])]
# %%
nsiml = 100
data = create_data_gen_np(
    plus_gen_list, minus_gen_list, nsiml=nsiml, z_plus_gen_large=plus_gen_list_large,
    z_minus_gen_large=minus_gen_list_large,
    estimators={"KLIEP Miss": est.kliep_miss_wrap,
                "KLIEP Naive": est.kliep_naive_wrap},
    miss_func_plus=miss_func_list, miss_func_minus=None,
    maxiter=100, alpha=0.1, delta=0.1, opt_type="BFGS")

# Do the same for known r
true_r_data = {key: [[] for iter in range(len(plus_gen_list))]
               for key in ["poweralpha", "classif"]}

for j in range(len(plus_gen_list)):
    for i in range(nsiml):
        c, classif = cutoff_bin(
            true_r, alpha=0.1, delta=0.1, newdata=minus_gen_list[j]())
        power = power_alpha_calc(classif, minus_gen_list_large[j](),
                                 plus_gen_list_large[j]())
        true_r_data["poweralpha"][j].append(power)
        true_r_data["classif"][j].append(c)

data["True R"] = true_r_data

with open('../results/simulated_results/vary_diffvar_misspec_'+str(nsiml)
          + 'sim_comp_diff='+str(diff)+'_torch.pkl', 'wb') as handle:
    pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

print("Multi-dimensional Asymmetric case completed")

# %%
# ##################### Incorrectly specified 3rd Setting ########################
lr = 0.7**(np.floor((np.arange(1000))/100)+1)
ns = np.arange(100, 1501, 100)
# Generate data generating procedures
cov_mat = torch.tensor([[1, 0], [0, 2]]).float()
diff = 1

plus_dist = dists.MultivariateNormal(torch.zeros(2), torch.eye(2))
minus_dist = dists.MultivariateNormal(torch.zeros(2)+diff, cov_mat)

plus_gen_list = [mv_sampler_creator(n, plus_dist)
                 for n in ns]
minus_gen_list = [mv_sampler_creator(n, minus_dist)
                  for n in ns]

plus_gen_list_large = [
    mv_sampler_creator(int(1e6), plus_dist) for n in ns]
minus_gen_list_large = [
    mv_sampler_creator(int(1e6), minus_dist) for n in ns]

p_0 = 0.8


# Create missing function
def miss_func(x):
    return torch.where(x[:, 1] > 0., p_0, 0.)


def true_r(x):
    return plus_dist.log_prob(x)-minus_dist.log_prob(x)


miss_func_list = [miss_func for iter in range(ns.shape[0])]
# %%
nsiml = 100
data = create_data_gen_np(
    plus_gen_list, minus_gen_list, nsiml=nsiml,
    z_plus_gen_large=plus_gen_list_large,
    z_minus_gen_large=minus_gen_list_large,
    estimators={"KLIEP Miss": est.kliep_miss_wrap,
                "KLIEP Naive": est.kliep_naive_wrap},
    miss_func_plus=miss_func_list, miss_func_minus=None,
    maxiter=100, alpha=0.1, delta=0.1, opt_type="BFGS")

# Do the same for known r
true_r_data = {key: [[] for iter in range(len(plus_gen_list))]
               for key in ["poweralpha", "classif"]}

for j in range(len(plus_gen_list)):
    for i in range(nsiml):
        c, classif = cutoff_bin(
            true_r, alpha=0.1, delta=0.1, newdata=minus_gen_list[j]())
        power = power_alpha_calc(classif, minus_gen_list_large[j](),
                                 plus_gen_list_large[j]())
        true_r_data["poweralpha"][j].append(power)
        true_r_data["classif"][j].append(c)

data["True R"] = true_r_data

with open('../results/simulated_results/vary_diffvar_misspec2_'+str(nsiml)
          + 'sim_comp_diff='+str(diff)+'_torch.pkl', 'wb') as handle:
    pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

print("Multi-dimensional Asymmetric case completed")

# %%
# ##################### Incorrectly specified 4th Setting ########################
# Mixture in plus class single in null
# Mixture probability varies
diff = 1
n = 500
mix_probs = [1, 0.95, 0.9, 0.8, 0.7, 0.6, 0.5]
# Define distributiosn
plus_dist_1 = dists.MultivariateNormal(torch.zeros(2), torch.eye(2))
plus_dist_2 = dists.MultivariateNormal(
    torch.tensor([2., 0.]), torch.eye(2))
minus_dist = dists.MultivariateNormal(
    torch.zeros(2)+diff, torch.eye(2))

# Set up list of samplers
plus_gen_list = [mv_mix_sampler_creator(n, plus_dist_1, plus_dist_2, mix_prob)
                 for mix_prob in mix_probs]
minus_gen_list = [mv_sampler_creator(n, minus_dist)
                  for i in range(len(mix_probs))]

# Set up list of large samplers
n_large = int(1e6)
plus_gen_list_large = [
    mv_mix_sampler_creator(n_large, plus_dist_1, plus_dist_2, mix_prob)
    for mix_prob in mix_probs]
minus_gen_list_large = [
    mv_sampler_creator(n_large, minus_dist)
    for i in range(len(mix_probs))]
# Create missing func
p_0 = 0.8


# Create missing function
def miss_func(x):
    return torch.where(x[:, 0] < 0., p_0, 0.)


def true_r(x, mix_prob):
    return torch.log(
        mix_prob*torch.exp(plus_dist_1.log_prob(x))
        + (1-mix_prob)*torch.exp(plus_dist_2.log_prob(x))
    )-minus_dist.log_prob(x)


miss_func_list = [miss_func for iter in range(len(mix_probs))]
# %%
nsiml = 100
data = create_data_gen_np(
    plus_gen_list, minus_gen_list, nsiml=nsiml,
    z_plus_gen_large=plus_gen_list_large,
    z_minus_gen_large=minus_gen_list_large,
    estimators={"KLIEP Miss": est.kliep_miss_wrap,
                "KLIEP Naive": est.kliep_naive_wrap},
    miss_func_plus=miss_func_list, miss_func_minus=None,
    maxiter=100, alpha=0.1, delta=0.1, opt_type="BFGS")

# Do the same for known r
true_r_data = {key: [[] for iter in range(len(plus_gen_list))]
               for key in ["poweralpha", "classif"]}

for j, mix_prob in enumerate(mix_probs):
    for i in range(nsiml):
        c, classif = cutoff_bin(
            lambda x: true_r(x, mix_prob),
            alpha=0.1, delta=0.1, newdata=minus_gen_list[j]())
        power = power_alpha_calc(classif, minus_gen_list_large[j](),
                                 plus_gen_list_large[j]())
        true_r_data["poweralpha"][j].append(power)
        true_r_data["classif"][j].append(c)

data["True R"] = true_r_data

with open('../results/simulated_results/vary_diffvar_misspec3_'+str(nsiml)
          + 'sim_comp_diff='+str(diff)+'_torch.pkl', 'wb') as handle:
    pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
print("Multi-dimensional Asymmetric case completed")

# %%
