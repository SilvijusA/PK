import numpy as np
import pandas as pd

# Example synthetic data
data = {
    'time': [0, 1, 2, 3, 4, 6, 8, 12],  # Time points in hours
    'concentration': [10.0, 8.0, 6.0, 5.0, 4.0, 2.5, 1.5, 0.5]  # Concentrations in mg/L
}
df = pd.DataFrame(data)

# Calculate AUC using the trapezoidal rule
def calculate_auc(time, concentration):
    auc = np.trapz(concentration, time)
    return auc

auc_0_t = calculate_auc(df['time'], df['concentration'])
print(f"AUC (0-t): {auc_0_t:.2f}")


#############
#Simulations#
#############

import numpy as np

# Parameters
D = 100  # Dose
V_true = 10  # True volume of distribution
k_true = 0.1  # True elimination rate constant
sigma = 1  # Measurement error

# Time points
t_obs = np.linspace(0, 24, 10)

# Generate synthetic data
C_true = (D / V_true) * np.exp(-k_true * t_obs)
C_obs = C_true + np.random.normal(0, sigma, size=t_obs.shape)


from scipy.stats import norm

def likelihood(V, k, C_obs, t_obs, D, sigma):
    C_pred = (D / V) * np.exp(-k * t_obs)
    return np.prod(norm.pdf(C_obs, loc=C_pred, scale=sigma))

def prior(V, k, mu_V=10, sigma_V=2, mu_k=0.1, sigma_k=0.02):
    prior_V = norm.pdf(V, loc=mu_V, scale=sigma_V)
    prior_k = norm.pdf(k, loc=mu_k, scale=sigma_k)
    return prior_V * prior_k


def posterior(V, k, C_obs, t_obs, D, sigma, mu_V=10, sigma_V=2, mu_k=0.1, sigma_k=0.02):
    return likelihood(V, k, C_obs, t_obs, D, sigma) * prior(V, k, mu_V, sigma_V, mu_k, sigma_k)


V_grid = np.linspace(5, 15, 100)
k_grid = np.linspace(0.05, 0.15, 100)
posterior_grid = np.zeros((len(V_grid), len(k_grid)))

for i, V in enumerate(V_grid):
    for j, k in enumerate(k_grid):
        posterior_grid[i, j] = posterior(V, k, C_obs, t_obs, D, sigma)

# Normalize the posterior
posterior_grid /= np.sum(posterior_grid)


V_samples, k_samples = np.meshgrid(V_grid, k_grid)
AUC_samples = D / (k_samples * V_samples)
AUC_posterior = np.sum(AUC_samples * posterior_grid)

print(f"Estimated AUC: {AUC_posterior}")



########
#Alternative
#############


import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

# Example data: time (hours) and plasma concentration (mg/L)
time = np.array([0, 1, 2, 4, 6, 8, 12, 24])
concentration = np.array([0, 5, 8, 6, 4, 2, 1, 0.5])

# Prior distribution parameters (mean and variance)
mu_prior = 50
sigma_prior = 5

# Likelihood function: assuming normal distribution of concentrations
def likelihood(AUC, time, concentration):
    predicted_concentration = AUC * np.exp(-time / 5)  # Example model
    return np.prod(stats.norm.pdf(concentration, predicted_concentration, 1))

# Posterior distribution
def posterior(AUC, time, concentration, mu_prior, sigma_prior):
    prior = stats.norm.pdf(AUC, mu_prior, sigma_prior)
    return likelihood(AUC, time, concentration) * prior

# Grid search for MAP estimate
AUC_values = np.linspace(0, 100, 1000)
posterior_values = [posterior(AUC, time, concentration, mu_prior, sigma_prior) for AUC in AUC_values]
AUC_MAP = AUC_values[np.argmax(posterior_values)]

print(f"MAP estimate of AUC: {AUC_MAP:.2f} mg·h/L")

# Plotting the posterior distribution
plt.plot(AUC_values, posterior_values, label='Posterior Distribution')
plt.axvline(AUC_MAP, color='r', linestyle='--', label=f'MAP Estimate: {AUC_MAP:.2f}')
plt.xlabel('AUC (mg·h/L)')
plt.ylabel('Posterior Probability')
plt.legend()
plt.show()


#Machine learning


import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Example data: time (hours) and plasma concentration (mg/L)
data = {
    'time': [0, 1, 2, 4, 6, 8, 12, 24],
    'concentration': [0, 5, 8, 6, 4, 2, 1, 0.5]
}
df = pd.DataFrame(data)

# Feature engineering: create features for the model
df['time_squared'] = df['time'] ** 2
df['time_log'] = np.log1p(df['time'])

# Target variable: AUC (for simplicity, using trapezoidal rule here)
df['auc'] = np.trapz(df['concentration'], df['time'])

# Prepare data for training
X = df[['time', 'time_squared', 'time_log']]
y = df['auc']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse:.2f}")

# Predict AUC for new data
new_data = pd.DataFrame({
    'time': [3, 5, 7, 9],
    'time_squared': [3**2, 5**2, 7**2, 9**2],
    'time_log': np.log1p([3, 5, 7, 9])
})
predicted_auc = model.predict(new_data)
print(f"Predicted AUC: {predicted_auc}")
