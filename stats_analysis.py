import pandas as pd
import numpy as np
from scipy.stats import binomtest
import numpy as np
import matplotlib.pyplot as plt

from lorenz_96 import lorenz_96
from lorenz_63 import lorenz_63
df_96 = pd.read_csv(rf'.\results\lorenz96_old.csv')

# for (activation, width), group_df in df.groupby(['ACTIVATION', 'HIDDEN_SIZE']):
#     x = len(group_df[group_df['MAE'] <0.07])
#     n = 50
#     print(f"activation: {activation}, width: {width}:")
#     result = binomtest(x, n)

#     ci = result.proportion_ci(
#         confidence_level=0.95,
#         method="wilson"
#     )
#     print(f"{result.statistic:3f} chance of models within 5% of L1")
#     print(f"CL: {ci.low:.3f}, {ci.high:.3f}")


# for (activation, width), group_df in df_96.groupby(['ACTIVATION', 'HIDDEN_SIZE']):
#     #x = len(group_df[group_df['MIN_SV3'] <0.5])
#     n = len(group_df)
#     print(f"activation: {activation}, width: {width}:")
#     print(n)

def find_diff(model_val: float,
              real_val:float) -> float:
    '''
    Calculates difference between real and mdoel values as a percentage from real values
    '''
    return np.abs(model_val - real_val)*100/(np.abs(real_val))
def stats(df: pd.DataFrame,
          column: str,
          real_val: float,
          percent: float) -> None:

    '''
    Calculates the % and Confidence intervals of each activation and width that fall wihtin a quantile of the true value.

    Inputs:
        df (pd.DataFrame): The dataframe to get stats from.
        column (str): The colummn/value we are getting stats from.
        real_val (float): The value to compare it against
        percent (float): The quantile to get proportion about.
    '''
    results = []
    lowers = []
    uppers = []
    betas = [0.5, 0.7, 1.0, 1.5, 2.0, 5.0]

    for (activation, width), group_df in df.groupby(['ACTIVATION', 'HIDDEN_SIZE']):
        diffs = group_df[column].apply(find_diff,real_val=real_val)
        x = len(group_df[diffs < percent])
        n = len(group_df)
        print(f"activation: {activation}, width: {width}:")
        result = binomtest(x, n)

        ci = result.proportion_ci(
            confidence_level=0.95,
            method="wilson"
        )
        print(f"{result.statistic:3f} chance of models within {percent}% of {column}")
        print(f"CL: {ci.low:.3f}, {ci.high:.3f}")

        results.append(result.statistic*100)
        lowers.append(ci.low*100)
        uppers.append(ci.high*100)


    fig, ax = plt.subplots(figsize=(10,8))

    x_pos = np.arange(len(betas))
    ax.plot(x_pos, results)
    ax.fill_between(
    x_pos,
    lowers,
    uppers,
    alpha=0.2,
    label='95% CI'
)
    ax.set_title("Percentage of Models with Minimum Gradient of $\hat\sigma_3$ Within 50% of True Value\n (95% C.I.)")
    ax.set_xlabel("Beta Value \ Maximum Gradient")
    ax.set_ylabel("% of Models")
    ax.set_xticks(x_pos)

    ax.set_xticklabels(
        [
            rf"$\beta$ = {b}" + "\n" + rf"Max grad = {b/4}"
            for b in betas
        ]
    )
    ax.tick_params(axis='x', labelsize=10)
    ax.set_ylim([0,105])
    fig.tight_layout()

    plt.show()


if __name__ == "__main__":
    #REAL L96 VALUES
    c = lorenz_96(8,8,0.01)
    x = np.array([8.01,8,8,8,8,8,8,8])

    svs_96 = c.find_svs(x=x, trajectory_steps=10000)

    mins_96 = np.min(svs_96, axis=0)
    grads_96 = (svs_96[1:,:] - svs_96[:-1,:])/c.dt
    min_grads_96 = np.min(grads_96, axis=0)

    ly_96 = c.find_lyapunov_spectrum(x=x)


    #REAL L63 VALUES
    c = lorenz_63()
    x = np.array([1,1,1])
    results_63 = c.find_lyapunov_spectrum(x=x)
    svs_63 = np.asarray(results_63['singular_values'])
    ly_63 = results_63['lyapunov_spectrum']

    mins_63 = np.min(svs_63, axis=0)
    grads_63 = (svs_63[1:,:] - svs_63[:-1,:])/c.dt
    min_grads_63 = np.min(grads_63, axis=0)


    stats(df=pd.read_csv(rf'.\betas.csv'), column = 'MIN_GRAD_SV3', real_val = min_grads_63[2], percent=50)


    