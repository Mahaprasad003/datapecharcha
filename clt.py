import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

def central_limit_theorem_app():
    """
    A Streamlit application to demonstrate the Central Limit Theorem.
    """
    st.set_page_config(layout="wide")

    # --- App Title and Description ---
    st.title("Central Limit Theorem Demonstration")
    st.markdown("""
    **How to use this app:**
    1.  **Choose a Population Distribution:** On the left, select a distribution for the parent population. Notice that some, like 'Exponential', are not bell-shaped at all.
    2.  **Set Parameters:** Adjust the parameters for the chosen distribution. Explanations for each parameter are provided below the sliders.
    3.  **Adjust Sample Size (n):** This is the crucial parameter. It's the number of data points in each individual sample we draw.
    4.  **Observe the Plots:**
        - The **left plot** shows the shape of the original population you're sampling from.
        - The **right plot** shows the distribution of the means of all the samples.
    
    **Your Goal:** See for yourself how the right plot becomes more bell-shaped as you increase the **Sample Size (n)**, no matter how the original population on the left looks!
    """)
    st.markdown("---")

    # --- Sidebar for User Inputs ---
    st.sidebar.header("Simulation Parameters")

    # Mapping from user-friendly names to scipy's internal names
    dist_mapping = {
        'Uniform': 'uniform',
        'Exponential': 'expon',
        'Binomial': 'binom',
        'Poisson': 'poisson'
    }
    
    dist_display_name = st.sidebar.selectbox(
        "Select a Distribution for the Population",
        list(dist_mapping.keys())
    )
    dist_name = dist_mapping[dist_display_name]

    # --- Distribution-specific parameters ---
    st.sidebar.markdown("---")
    st.sidebar.subheader(f"Parameters for {dist_display_name} Dist.")
    
    population_params = {}
    if dist_display_name == 'Uniform':
        st.sidebar.markdown("*A distribution where every outcome is equally likely.*")
        low = st.sidebar.slider("Lower Bound (a)", -10.0, 10.0, 0.0, 0.1)
        st.sidebar.markdown("> **What is this?** The minimum possible value in the distribution.")
        high = st.sidebar.slider("Upper Bound (b)", low + 0.1, low + 20.0, 10.0, 0.1)
        st.sidebar.markdown("> **What is this?** The maximum possible value in the distribution. Any number between `a` and `b` has an equal chance of being chosen.")
        population_params = {'loc': low, 'scale': high - low}
        theo_mean = (low + high) / 2
        theo_var = ((high - low) ** 2) / 12

    elif dist_display_name == 'Exponential':
        st.sidebar.markdown("*Describes the time between events in a process. It's heavily skewed.*")
        scale = st.sidebar.slider("Scale (β)", 0.1, 10.0, 2.0, 0.1)
        st.sidebar.markdown("> **What is this?** The average time between events. For example, if the average time between customer arrivals is 2 minutes, the scale is 2. A larger scale stretches the distribution out.")
        population_params = {'scale': scale}
        theo_mean = scale
        theo_var = scale ** 2

    elif dist_display_name == 'Binomial':
        st.sidebar.markdown("*Represents the number of 'successes' in a fixed number of trials.*")
        n_trials = st.sidebar.slider("Number of Trials (n_trials)", 1, 100, 20, 1)
        st.sidebar.markdown("> **What is this?** The total number of times an experiment is run (e.g., flipping a coin 20 times).")
        p_success = st.sidebar.slider("Probability of Success (p)", 0.0, 1.0, 0.5, 0.01)
        st.sidebar.markdown("> **What is this?** The chance of a single trial being a 'success' (e.g., the probability of getting heads, which is 0.5 for a fair coin).")
        population_params = {'n': n_trials, 'p': p_success}
        theo_mean = n_trials * p_success
        theo_var = n_trials * p_success * (1 - p_success)

    elif dist_display_name == 'Poisson':
        st.sidebar.markdown("*Represents the number of events in a fixed interval.*")
        lam = st.sidebar.slider("Rate (λ)", 0.1, 20.0, 5.0, 0.1)
        st.sidebar.markdown("> **What is this?** The average number of events in an interval. For example, if a website gets an average of 5 clicks per minute, the rate (λ) is 5.")
        population_params = {'mu': lam}
        theo_mean = lam
        theo_var = lam

    # --- Sample size and number of samples ---
    st.sidebar.markdown("---")
    st.sidebar.subheader("Sampling Parameters")
    sample_size = st.sidebar.slider(
        "Sample Size (n)",
        min_value=1,
        max_value=500,
        value=30,
        help="The number of data points drawn from the population for each sample. This is the most important parameter for the CLT!"
    )
    st.sidebar.markdown("> **What is this?** The number of individual data points in each sample we take. The CLT's effect becomes stronger with a larger sample size. A common rule of thumb is n > 30.")
    
    num_samples = st.sidebar.slider(
        "Number of Samples",
        min_value=100,
        max_value=10000,
        value=1000,
        step=100,
        help="The number of times we draw a sample of size 'n'. A higher number gives a smoother histogram."
    )
    st.sidebar.markdown("> **What is this?** The total number of times we perform the sampling experiment. Each time, we take `n` items and calculate their mean. More samples will give a smoother histogram on the right.")


    # --- Data Generation ---
    @st.cache_data
    def generate_data(dist, params, pop_size, n_samples, samp_size):
        # Generate a large population to sample from
        population = getattr(stats, dist).rvs(size=pop_size, **params)
        # Generate samples and calculate their means
        sample_means = [np.mean(np.random.choice(population, size=samp_size, replace=True)) for _ in range(n_samples)]
        return population, sample_means

    population, sample_means = generate_data(dist_name, population_params, 100000, num_samples, sample_size)

    # --- Visualization ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    plt.style.use('seaborn-v0_8-whitegrid')

    # Plot 1: Population Distribution
    ax1.hist(population, bins=50, density=True, alpha=0.7, label='Population Data', color='skyblue', edgecolor='black')
    ax1.axvline(theo_mean, color='red', linestyle='--', linewidth=2, label=f'Theoretical Mean: {theo_mean:.2f}')
    ax1.set_title(f'1. Population Distribution ({dist_display_name})', fontsize=16)
    ax1.set_xlabel('Value', fontsize=12)
    ax1.set_ylabel('Density', fontsize=12)
    ax1.legend()

    # Plot 2: Distribution of Sample Means
    ax2.hist(sample_means, bins=50, density=True, alpha=0.7, label='Sample Means', color='salmon', edgecolor='black')
    mu_sample_means = np.mean(sample_means)
    std_sample_means = np.std(sample_means)
    x = np.linspace(min(sample_means), max(sample_means), 100)
    p = stats.norm.pdf(x, mu_sample_means, std_sample_means)
    ax2.plot(x, p, 'k', linewidth=2, label='Fitted Normal Curve')
    ax2.axvline(mu_sample_means, color='blue', linestyle='--', linewidth=2, label=f'Mean of Means: {mu_sample_means:.2f}')
    ax2.set_title('2. Distribution of the Sample Means', fontsize=16)
    ax2.set_xlabel('Sample Mean Value', fontsize=12)
    ax2.set_ylabel('Density', fontsize=12)
    ax2.legend()
    
    fig.tight_layout(pad=3.0)
    st.pyplot(fig)

    # --- Explanations and Statistics ---
    st.markdown("---")
    st.header("What's Happening Here? (The Magic of CLT)")
    st.markdown(f"""
    Even if the population distribution is skewed (e.g., exponential) or flat (e.g., uniform), the distribution of sample means becomes approximately normal as sample size increases. This is the Central Limit Theorem: regardless of the population's shape, the sampling distribution of the mean approaches a normal, symmetric, bell-shaped distribution
    """)

    st.markdown("---")
    st.header("Statistical Comparison")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Original Population")
        st.markdown("These are the theoretical stats for the parent distribution on the left.")
        st.metric("Population Mean (μ)", f"{theo_mean:.3f}")
        st.metric("Population Std Dev (σ)", f"{np.sqrt(theo_var):.3f}")

    with col2:
        st.subheader("Sample Means")
        st.markdown("These are the *actual* stats calculated from our simulation on the right.")
        st.metric("Mean of Sample Means", f"{mu_sample_means:.3f}")
        st.metric("Std Dev of Sample Means", f"{std_sample_means:.3f}")

    with col3:
        st.subheader("CLT Predictions")
        st.markdown("This is what the CLT predicts the stats for the sample means should be.")
        clt_mean = theo_mean
        clt_std_dev = np.sqrt(theo_var) / np.sqrt(sample_size)
        st.metric("Predicted Mean (μ)", f"{clt_mean:.3f}")
        st.metric("Predicted Std Dev (σ/√n)", f"{clt_std_dev:.3f}")
        st.markdown("**Key Insight:** Notice how the *Mean of Sample Means* is very close to the *Population Mean*, and the *Std Dev of Sample Means* is very close to the *CLT Predicted Std Dev*. This confirms the theorem!")

if __name__ == '__main__':
    central_limit_theorem_app()