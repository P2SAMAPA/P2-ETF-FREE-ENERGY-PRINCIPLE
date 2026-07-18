import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.optimize import minimize

def compute_composite_macro_factor(macro_df):
    """Compute composite macro factor from all macro variables."""
    if len(macro_df) < 2:
        return np.ones(len(macro_df)) * 0.5
    scaler = StandardScaler()
    macro_scaled = scaler.fit_transform(macro_df)
    pca = PCA(n_components=1)
    factor = pca.fit_transform(macro_scaled).flatten()
    factor = (factor - factor.min()) / (factor.max() - factor.min() + 1e-8)
    return factor

def agent_generative_model(state, dim=8):
    """
    An agent's internal generative model: maps hidden states to observations.
    This is a simplified neural network-like mapping.
    """
    if len(state) < 3:
        return np.zeros(dim)
    # Use moments and features of the state as the generative representation
    features = np.array([
        np.mean(state),
        np.std(state),
        np.percentile(state, 25),
        np.percentile(state, 50),
        np.percentile(state, 75),
        np.max(np.abs(np.diff(state))),
        np.mean(np.abs(np.diff(state))),
        np.std(np.diff(state))
    ])
    if len(features) < dim:
        features = np.pad(features, (0, dim - len(features)), 'constant')
    else:
        features = features[:dim]
    return features

def variational_free_energy(returns, macro_factor, n_agents=10, generative_dim=8, iterations=30):
    """
    Compute the variational free energy of the market system.
    Lower free energy = more coherent market belief / stable regime.
    """
    if len(returns) < n_agents + 5:
        return 0.0
    # Split returns into agents (time-lagged subpopulations)
    agent_states = []
    for i in range(n_agents):
        if len(returns) > i:
            agent_states.append(returns[i::n_agents])
    if len(agent_states) < 2:
        return 0.0
    # Each agent has a generative model of the world
    generative_models = []
    for state in agent_states:
        if len(state) > 5:
            gen = agent_generative_model(state, generative_dim)
            generative_models.append(gen)
    if len(generative_models) < 2:
        return 0.0
    # Compute the "belief" of each agent (posterior over hidden states)
    beliefs = []
    for i, gen in enumerate(generative_models):
        # Belief is the similarity between an agent's model and the average model
        avg_model = np.mean(generative_models, axis=0)
        if np.linalg.norm(avg_model) > 0:
            belief = np.dot(gen, avg_model) / (np.linalg.norm(gen) * np.linalg.norm(avg_model))
        else:
            belief = 0.0
        beliefs.append(belief)
    # Average belief (coherence)
    avg_belief = np.mean(beliefs)
    # Surprise = negative log likelihood of observations under the generative model
    # We use the variance of beliefs as a proxy for surprise
    surprise = np.std(beliefs)
    # Free energy = surprise - entropy (simplified)
    # In active inference, free energy = surprise + KL divergence
    # Here we use a simplified form: free energy = surprise - coherence
    free_energy = surprise - avg_belief
    # Scale by macro factor
    free_energy = free_energy * (1 + macro_factor * 0.5)
    return float(free_energy)

def free_energy_score(returns, macro_df, n_agents=10, generative_dim=8, iterations=30):
    """
    Compute per-ETF free energy score.
    Higher score = more coherent market belief / stable regime.
    """
    if len(returns) < 15 or macro_df is None or len(macro_df) < 15:
        return 0.0
    # Align lengths
    min_len = min(len(returns), len(macro_df))
    returns = returns[:min_len]
    macro_df = macro_df.iloc[:min_len]
    # Remove NaN
    mask = ~(np.isnan(returns) | np.isnan(macro_df).any(axis=1))
    returns = returns[mask]
    macro_df = macro_df[mask]
    if len(returns) < 15:
        return 0.0
    # Compute macro factor
    macro_factor = compute_composite_macro_factor(macro_df)[-1]
    # Compute free energy
    fe = variational_free_energy(returns, macro_factor, n_agents, generative_dim, iterations)
    # Invert so that lower free energy -> higher score
    # We want high score = coherent market belief
    score = 1.0 / (1.0 + fe)
    return float(score)
