import pandas as pd
import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt

# ===========================================================
# 1. DATA LOADING & PRE-PROCESSING
# ===========================================================
def run_tafita_analysis(file_path):
    # Load dataset
    df = pd.read_csv(file_path)
    
    # Identify action columns (TaRL activities)
    action_cols = [c for c in df.columns if 'freq_pos1::' in c]
    states = [0, 1, 2, 3, 4]  # 0:Beg, 1:1CH, 2:2CH, 3:3A, 4:3B (Mastery)

    # ===========================================================
    # 2. EMPIRICAL TRANSITION MATRICES ESTIMATION
    # ===========================================================
    def get_action_matrices(data, actions):
        catalogue = {}
        for a in actions:
            P = np.zeros((5, 5))
            for s in states[:-1]:
                # Filter students at state 's' who practiced activity 'a'
                subset = data[(data['Pretest'] == s) & (data[a] > 0)]
                if len(subset) > 3:  # Reliability threshold
                    counts = subset['Final'].value_counts(normalize=True)
                    for next_s, prob in counts.items():
                        P[s, int(next_s)] = prob
                else:
                    P[s, s] = 1.0  # Default to stagnation (stochastic friction)
            P[4, 4] = 1.0  # Absorbing state
            catalogue[a] = P
        return catalogue

    matrices = get_action_matrices(df, action_cols)

    # ===========================================================
    # 3. MDP RESOLUTION (POLICY OPTIMIZATION)
    # ===========================================================
    def solve_bellman(matrices):
        V = np.zeros(5) 
        policy = {}
        # Value Iteration to minimize expected time to mastery
        for _ in range(100):
            V_new = np.zeros(5)
            for s in states[:-1]:
                # Bellman Equation: 1 hour + expected future value
                q_values = {a: 1 + np.dot(matrices[a][s], V) for a in matrices}
                best_a = min(q_values, key=q_values.get)
                V_new[s] = q_values[best_a]
                policy[s] = best_a
            V = V_new
        return policy, V

    opt_policy, expected_times = solve_bellman(matrices)

    # ===========================================================
    # 4. MONTE-CARLO SIMULATION & SURVIVAL ANALYSIS
    # ===========================================================
    def simulate_cohort(policy_type, matrices, n=2000):
        steps = []
        for _ in range(n):
            s, t = 0, 0
            while s < 4 and t < 150:
                if policy_type == 'optimal':
                    a = opt_policy[s]
                else:
                    # Simulation of a random/standard mix
                    a = np.random.choice(action_cols)
                s = np.random.choice(states, p=matrices[a][s])
                t += 1
            steps.append(t)
        return steps

    res_opt = simulate_cohort('optimal', matrices)
    res_std = simulate_cohort('standard', matrices)

    # ===========================================================
    # 5. VISUALIZATION: SURVIVAL CURVES
    # ===========================================================
    max_t = 120
    time_axis = np.arange(0, max_t)
    surv_std = [np.mean(np.array(res_std) > t) for t in time_axis]
    surv_opt = [np.mean(np.array(res_opt) > t) for t in time_axis]

    plt.figure(figsize=(10, 6))
    plt.step(time_axis, surv_std, label='Standard Policy (Empirical mix)', color='red', alpha=0.7)
    plt.step(time_axis, surv_opt, label='Optimal Policy $\pi^*$ (Spectral Min)', color='green', linewidth=2)
    plt.fill_between(time_axis, surv_opt, surv_std, color='gray', alpha=0.1)
    plt.title('Survival Analysis: Probability of Non-Mastery over Time')
    plt.xlabel('Instructional Hours')
    plt.ylabel('P(T > t)')
    plt.grid(alpha=0.2)
    plt.legend()
    plt.show()

    # ===========================================================
    # 6. LATEX TABLE GENERATION
    # ===========================================================
    print("\n% --- LATEX OPTIMAL POLICY TABLE ---")
    print("\\begin{table}[ht]\n\\centering\n\\begin{tabular}{llc}")
    print("\\hline\nLevel & Optimal Activity ($\\pi^*$) & Local Friction ($\\rho_{ii}$) \\\\ \\hline")
    for s, a in opt_policy.items():
        clean_name = a.split('::')[1].replace('_', ' ')
        friction = matrices[a][s, s]
        print(f"L{s+1} & {clean_name} & {friction:.2f} \\\\")
    print("\\hline\n\\end{tabular}\n\\end{table}")

# Execute
run_tafita_analysis('data_maths_elysa.csv')