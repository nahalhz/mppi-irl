"""Model a skid steer robot on simple 2D grid"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import truncnorm
from scipy.stats import norm
from scipy.ndimage import gaussian_filter
import time as timer

# for reproducing the risk map
SEED = 2
np.random.seed(SEED)
RNG = np.random.RandomState(SEED)

### PARAMETERS
# grid
# total grid size - (10 m) x (10 m) assume square for now
GRID_CELL_SIZE = 0.1                  # metres
GRID_L = 10                           # metres
RISK_MAP_DIM = (int) (GRID_L/GRID_CELL_SIZE)
GRID_RANGE = [0, GRID_L]              # for clipping

# skid steer robot
L = 0.5                               # wheel base distance (metres)
r = 0.1                               # wheel radius (metres)
INITIAL_STATE = np.array([9.5,5.8,np.pi])
GOAL_STATE = np.array([5.9,0.3,0]) # goal

# control constraints
V_MAX = 1.0                           # max allowable linear velocity (m/s)
W_MAX = V_MAX/r                       # convert to max angular velocity (rad/s)

# simulation params
SIM_UNTIL_GOAL = True                 # when true overrule T_sim
T = 10                                # horizon (look ahead) time in s
T_sim = 100                           # total simulation of robot in s
# 0.1 for now to make computation easier
dt = 0.01 # 0.01 # timestep in s

### functions
# from skid_steer.py
def dynamics(state, control, r=r, L=L):
    """Simulate skid steer dynamics for one time step  state: np.ndarray stores x, y, theta  control : np.ndarray stores u_left and u_right, velocity commands for wheels."""
    x, y, theta = state
    u_l, u_r = control
    
    dx = r * (u_l + u_r) * np.cos(theta) / 2
    dy = r * (u_l + u_r) * np.sin(theta) / 2
    dtheta = r * (u_r - u_l) / L
        
    return np.array([dx, dy, dtheta])

# should add weightings for risk/distance
def evaluate_cost(x, u, goal_state, risk_map):
    cost=0
    # weighting to avoid high-risk above all else
    # minimal control weights may overshoot/spin out
    # W[0]: weight on risk
    # W[1]: weight on per-state distance from goal
    # W[2]: weight on control effort
    # W[3]: weight on terminal state distance from goal
    W = [1, 0, 5.0/600, 25]
    # compute risk over the path
    for ind, state in enumerate(x):
        # pos in metres -> grid indices for map
        # clip to handle boundary cases
        i = int(np.clip(np.floor(state[0]/GRID_CELL_SIZE), 0, RISK_MAP_DIM-1))
        j = int(np.clip(np.floor(state[1]/GRID_CELL_SIZE), 0, RISK_MAP_DIM-1))
        cost_risk = risk_map[i,j] # w[0] * risk_map[i,j] in case of static alpha

        # err based on distance from goal (x,y only, ignoring theta)
        cost_dist = np.linalg.norm(goal_state[:2] - state[:2])

        # to penalize high velocity --> overshoot
        cost_controls = np.sum(np.square(u[ind,:]))

        # per-state cost
        cost += cost_risk + W[1]*cost_dist + W[2]*cost_controls

    # penalize final state
    total_cost = cost + W[3] * np.linalg.norm(goal_state[:2] - x[-1,:2])
    return total_cost

# x0 - initial state
# u - control sequence (T/dt x 2)
def simulate(x0, u, dt=dt, T=T):
    
    # x[i] is state (x,y,theta) at timestep i
    x = np.zeros((int(np.ceil(T/dt)), 3))
    x[0,:] = x0

    # iterate over all timesteps in sim
    for t_idx in range(1,int(np.ceil(T/dt))):
        dx_t = dynamics(x[t_idx-1,:], u[t_idx-1,:])
        # next state based on dynamics
        x[t_idx,:] = x[t_idx-1,:] + dt*dx_t
        # clip robot into grid
        x[t_idx,0] = np.clip(x[t_idx,0], GRID_RANGE[0], GRID_RANGE[1])
        x[t_idx,1] = np.clip(x[t_idx,1], GRID_RANGE[0], GRID_RANGE[1])
    
    # full trajectory over T using this control sequence
    return x

def generate_cost_map(x_dim, y_dim):
    # sample each cell of cost map from a normal dist
    # this could be cost of traversing terrain at that cell etc
    # arbitrary choices for mean and variance right now
    mu = 0.7
    sigma = 1.0
    cost_map = np.random.normal(loc=mu, 
                                scale=sigma,
                                size=(x_dim,y_dim))
    
    # custom safe zone -- testing
    for i in range(60,86):
        for j in range(54,68):
            cost_map[i,j] = 0.4

    # convolve with gaussian kernel for smoothness
    # arbitrary spread for now
    g_sigma = 5.0
    cost_map = gaussian_filter(cost_map, sigma=g_sigma)

    # normalize on [0,1]
    cost_map = (cost_map - np.min(cost_map)) / (np.max(cost_map) - np.min(cost_map))

    return cost_map

def cvar_risk_map(x_dim, y_dim, beta, cost_map):
    ## CVaR (conditional value at risk)
    # ref - https://gdmarmerola.github.io/risk-aware-bandits/
    grid = np.linspace(0.001, 0.999, 1001)
    
    # computing CVaR for each cell based on underlying cost map
    risk_map_cvar = np.zeros_like(cost_map)
    for i in range(x_dim):
        for j in range(y_dim):
            # map high risk from map directly as the mean
            cell_mean = cost_map[i, j]

            # create different tails by letting higher risk mean more uncertainty
            # i.e larger risk_map[i,j] -> larger cell_sigma
            # assuming some kind of linear scaling of uncertainty with cost
            scaling_factor = 2
            baseline_uncertainty = 0.05
            cell_sigma = baseline_uncertainty + cost_map[i, j] * scaling_factor

            # create the normal distribution for cell
            dist = norm(cell_mean, cell_sigma)
            percentiles = dist.ppf(grid)

            # CVaR map entry is now the average cost of worst beta % outcomes
            # since using cost and not gain looking at upper tail
            risk_map_cvar[i, j] = percentiles[grid > (1-beta)].mean()

    return risk_map_cvar

# one step of optimization using the cross-entropy method (CEM)
def cem_step(x0, goal_state, risk_map, mu_init, sigma_init):
    N_SAMPLES = 100      # num of control sequences to sample
    N_ELITES = 20        # num of best samples to keep
    N_ITERATIONS = 2     # num times to refine the search per MPC step
    # damping params
    ALPHA_CEM = 0.3          # smoothing factor on old dist
    SIGMA_MIN = 0.5
    
    # start with initial distribution
    mu = mu_init
    sigma = sigma_init

    for i in range(N_ITERATIONS):
        # standardize the limits relative to the current mu and sigma
        a_i = (-W_MAX - mu) / sigma
        b_i = (W_MAX - mu) / sigma

        # sample control sequences
        U_samples = truncnorm.rvs(
            a=a_i, 
            b=b_i, 
            loc=mu, 
            scale=sigma, 
            size=(N_SAMPLES, int(np.ceil(T/dt)), 2)
        )
        
        # evaluate costs
        costs = np.zeros(N_SAMPLES)
        for i in range(N_SAMPLES):
            # pass the sampled sequence U_samples[i] to the simulation
            X_i = simulate(x0, U_samples[i])
            costs[i] = evaluate_cost(X_i, U_samples[i], goal_state, risk_map)
            
        # select best performers
        elite_indices = np.argsort(costs)[:N_ELITES]
        elite_samples = U_samples[elite_indices]
        
        ## update distribution
        # update mean (mu) and standard deviation (sigma) based on elites
        mu_new = np.mean(elite_samples, axis=0)   # (N, 2)
        sigma_new = np.std(elite_samples, axis=0) # (N, 2)
        
        # smoothing
        mu = ALPHA_CEM * mu + (1 - ALPHA_CEM)*mu_new
        sigma = ALPHA_CEM * sigma + (1 - ALPHA_CEM)*sigma_new
        
        # prevent sigma from collapsing to zero
        sigma = np.maximum(sigma, SIGMA_MIN) 
        
    return mu, sigma

def plot_trajectory(x, goal_state, risk_map):
    fig, ax = plt.subplots(figsize=(10,10))

    # plot risk map as heat map (for when continuous)
    im = ax.imshow(
        risk_map.T, 
        origin='lower', 
        extent=[GRID_RANGE[0], GRID_RANGE[1], GRID_RANGE[0], GRID_RANGE[1]],
        cmap='Reds',
        alpha=0.6
    )
    plt.colorbar(im, ax=ax, label='CVaR Cost')

    # plot trajectory
    ax.plot(x[:,0], x[:,1], 'b-o', linewidth=2, markersize=4)
    ax.plot(x[0,0], x[0,1], 'go', markersize=8, label='Start')
    ax.plot(x[-1,0], x[-1,1], 'ro', markersize=8, label='End')
    ax.plot(goal_state[0], goal_state[1], 'mo', markersize=8, label='Goal')

    # plot headings
    num_arrows = 10
    marker_size = 100  # size of triangles
    for i in range(0, len(x), len(x) // num_arrows):
        theta = x[i,2]
        ax.scatter(x[i,0], x[i,1], marker=(3, 0, np.degrees(theta) - 90), 
                    s=marker_size, c='red', alpha=0.8, zorder=5)

    # grid for grid map
    ax.grid(which='minor',color='#AAAAAA', linestyle='-', linewidth=0.5, alpha=0.7)
    ax.grid(which='major',color='#AAAAAA', linestyle='-', linewidth=0.5, alpha=0.7)
    ax.set_xlim(GRID_RANGE[0], GRID_RANGE[1])
    ax.set_ylim(GRID_RANGE[0], GRID_RANGE[1])
    ax.set_xticks(np.arange(GRID_RANGE[0], GRID_RANGE[1]+GRID_CELL_SIZE, GRID_CELL_SIZE), minor=True)
    ax.set_yticks(np.arange(GRID_RANGE[0], GRID_RANGE[1]+GRID_CELL_SIZE, GRID_CELL_SIZE), minor=True)

    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_title('Skid-steer robot traversal')
    ax.legend()
    ax.set_aspect('equal', adjustable='box')
    plt.show()

def main():
    # CVaR computed as average cost from only the most costly (BETA * 100%) outcomes -- tail risk
    # BETA -> 1 - do not care about risk returns average cost over entire risk dist
    # BETA -> 0 - risk-averse returns cost for absolute worst case outcome
    beta_CVaR = 0.1

    # grid - (x,y) = (0,0) bottom left
    cost_map = generate_cost_map(RISK_MAP_DIM, RISK_MAP_DIM)
    risk_map = cvar_risk_map(RISK_MAP_DIM, RISK_MAP_DIM, beta_CVaR, cost_map)

    ## optimize controls
    current_state = INITIAL_STATE
    executed_path = [current_state.copy()]

    # initial guess - zero control mean wide standard deviation for high initial exploration
    mu_control = np.zeros((int(np.ceil(T/dt)), 2))
    sigma_control = np.ones((int(np.ceil(T/dt)), 2)) * W_MAX * 0.5 # start with half the max range

    ## SIMULATION
    max_steps = int(np.ceil(T_sim/dt))
    if SIM_UNTIL_GOAL:
        print(f'Sim until robot reaches goal.')
        print(f"Starting MPC sim with Max U={W_MAX} rad/s.")
    else:
        print(f'Sim time is {T_sim} seconds')
        print(f"Starting MPC sim with Max U={W_MAX} rad/s. Max {max_steps} steps")

    start_time = timer.time()
    curr_step = 0
    check_stuck_iters = 100

    # tolerance w.r.t grid cell
    eps = GRID_CELL_SIZE/2

    ### execution loop
    while True:
        # break if max # steps is set otherwise break when goal is reached
        if not SIM_UNTIL_GOAL and curr_step >= max_steps:
            break

        # stuck detection - check if "no progress" made between current state and past states
        # if so increase beta to allow riskier decisions to reach the goal
        if (curr_step % check_stuck_iters == 0 and curr_step >= check_stuck_iters):
            if (np.abs(current_state[0] - executed_path[curr_step-check_stuck_iters][0]) <= eps*2):
                if (np.abs(current_state[1] - executed_path[curr_step-check_stuck_iters][1]) <= eps*2):
                    old_beta = beta_CVaR
                    # increase cap at 1
                    beta_CVaR = np.minimum(beta_CVaR + 0.1, 1)
                    if (beta_CVaR > old_beta):
                        print(f'Beta updated to {beta_CVaR:.1f}')
                        # update risk map
                        risk_map = cvar_risk_map(RISK_MAP_DIM, RISK_MAP_DIM, beta_CVaR, cost_map)

                    ### shouldn't reach this case unless terminal weight is too low in cost function
                    else:
                        print(f'Robot stuck even at max beta={beta_CVaR}')
                        break
    
        # optimizer
        best_u_plan, new_sigma = cem_step(
            x0=current_state,
            goal_state=GOAL_STATE,
            risk_map=risk_map,
            mu_init=mu_control,
            sigma_init=sigma_control
        )
        
        # execute
        optimal_control = best_u_plan[0,:] # execute only the first control
        dx_applied = dynamics(current_state, optimal_control)
        current_state = current_state + dt * dx_applied
        # clip robot into grid
        current_state[0] = np.clip(current_state[0], GRID_RANGE[0], GRID_RANGE[1])
        current_state[1] = np.clip(current_state[1], GRID_RANGE[0], GRID_RANGE[1])
        executed_path.append(current_state.copy())
        
        # shift the optimal plan for the next step
        # the best plan from this step becomes the starting guess for the next
        mu_control = np.roll(best_u_plan, -1, axis=0)
        # predict the last step is similar to the second-to-last
        mu_control[-1, :] = mu_control[-2, :] 
        
        sigma_control = np.roll(new_sigma, -1, axis=0)
        sigma_control[-1, :] = new_sigma[-2, :]
        
        # logging
        distance_to_goal = np.linalg.norm(current_state[:2] - GOAL_STATE[:2])
        map_i = int(np.floor(current_state[0]/GRID_CELL_SIZE))
        map_j = int(np.floor(current_state[1]/GRID_CELL_SIZE))

        # logging every 10 steps
        if (curr_step % 10 == 0):
            print(f"Step {curr_step}: Pos=({current_state[0]:.2f}, {current_state[1]:.2f}), Map[{map_i}][{map_j}], "
                f"Goal Dist={distance_to_goal:.2f}m, "
                f"U_applied=({optimal_control[0]:.2f}, {optimal_control[1]:.2f})")

        # final log
        if distance_to_goal < eps:
            print(f"Step {curr_step}: Pos=({current_state[0]:.2f}, {current_state[1]:.2f}), Map[{map_i}][{map_j}], "
              f"Goal Dist={distance_to_goal:.2f}m, "
              f"U_applied=({optimal_control[0]:.2f}, {optimal_control[1]:.2f})")
            print("Goal reached")
            break

        # increment step
        curr_step += 1

    end_time = timer.time()
    elapsed_time = end_time - start_time
    print(f'Elapsed time for simulation: {elapsed_time:.2f} s.')

    ## PLOT
    path_arr = np.array(executed_path)
    plot_trajectory(path_arr, GOAL_STATE, risk_map)

if __name__ == "__main__":
    main()
