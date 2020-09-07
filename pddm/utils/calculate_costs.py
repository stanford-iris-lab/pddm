import numpy as np
from scipy.special import softmax



def acc_costs_per_step(pt, prev_pt, costs, actions, dones, reward_func):
    """Accumulative costs per step."""
    step_rews, step_dones = reward_func(pt, actions)
    dones = np.logical_or(dones, step_dones)
    costs[dones > 0] += 500
    costs[dones == 0] -= step_rews[dones == 0]
    return costs, dones

def step_costs(states_t, actions_t, dones, reward_func):
    costs = np.zeros(dones.shape)
    step_rews, step_dones = reward_func(states_t, actions_t)
    dones = np.logical_or(dones, step_dones)
    costs[dones > 0] += 500
    costs[dones == 0] -= step_rews[dones == 0]

    # print(f" states_t.shape: {states_t.shape} \n actions_t.shape: {actions_t.shape} \n dones.shape: {dones.shape} \n costs.shape: {costs.shape} \n step_rews.shape: {step_rews.shape} \n step_dones.shape: {step_dones.shape}")
    return costs, dones



def calculate_costs(
    resulting_states_list, actions, reward_func, evaluating, take_exploratory_actions, disc_logits=None
):
    """Rank various predicted trajectories (by cost)

    Args:
        resulting_states_list :
            predicted trajectories
            [ensemble_size, horizon+1, N, statesize]
        actions :
            the actions that were "executed" in order to achieve the predicted trajectories
            [N, h, acsize]
        reward_func :
            calculates the rewards associated with each state transition in the predicted trajectories
        evaluating :
            determines whether or not to use model-disagreement when selecting which action to execute
            bool
        take_exploratory_actions :
            determines whether or not to use model-disagreement when selecting which action to execute
            bool

    Returns:
        cost_for_ranking : cost associated with each candidate action sequence [N,]
    """

    ensemble_size = len(resulting_states_list)
    tiled_actions = np.tile(actions, (ensemble_size, 1, 1))

    ###########################################################
    ## some reshaping of the predicted trajectories to rate
    ###########################################################

    N = len(resulting_states_list[0][0])

    # resulting_states_list is [ensSize, H+1, N, statesize]
    resulting_states = []
    for timestep in range(len(resulting_states_list[0])):  # loops over H+1
        all_per_timestep = []
        for entry in resulting_states_list:  # loops over ensSize
            all_per_timestep.append(entry[timestep])
        all_per_timestep = np.concatenate(all_per_timestep)  # [ensSize*N, statesize]
        resulting_states.append(all_per_timestep)
    # resulting_states is now [H+1, ensSize*N, statesize]

    ###########################################################
    ## calculate costs associated with each predicted trajectory
    ######## treat each traj from each ensemble as just separate trajs
    ###########################################################

    # init vars for calculating costs
    costs = np.zeros((N * len(resulting_states_list),))
    prev_pt = resulting_states[0]
    dones = np.zeros((N * len(resulting_states_list),))

    # accumulate cost over each timestep
    for pt_number in range(len(resulting_states_list[0]) - 1):

        # array of "current datapoint" [ensemble_size x N, state_size]
        pt = resulting_states[pt_number + 1]
        # update cost at the next timestep of the H-step rollout
        actions_per_step = tiled_actions[:, pt_number]
        costs, dones = acc_costs_per_step(
            pt, prev_pt, costs, actions_per_step, dones, reward_func
        )
        # update
        prev_pt = np.copy(pt)

    original_costs = np.copy(costs)

    # non_accumulated costs
    horizon = len(resulting_states_list[0]) - 1
    dones = np.zeros((N * len(resulting_states_list),))
    model_ens_size = len(resulting_states_list)
    stepwise_costs = []
    # resulting_state: [horizon + 1, model_ens_size * N, statesize]
    # step_costs: [horizon, model_ens_size * N]
    # stepwise_costs.shape: [horizon, model_ens_size * N]

    for t in range(horizon):
        states_t = resulting_states[t + 1] # [model_ens_size * N, statesize]
        actions_t = tiled_actions[:, t]
        costs_t, dones = step_costs(states_t, actions_t, dones, reward_func)
        stepwise_costs.append(costs_t)

    stepwise_costs = np.array(stepwise_costs)

    assert np.allclose(np.sum(stepwise_costs, axis = 0), original_costs), f"np.sum(stepwise_costs, axis = 0) = {np.sum(stepwise_costs, axis = 0)} != {costs} = costs"



    ###########################################################
    ## assigns costs associated with each predicted trajectory
    ####### need to consider each ensemble separately again
    ####### perform ranking based on either
    # "mean costs" over ensemble predictions (for a given action sequence A)
    # or
    # "model disagreement" over ensemble predictions (for a given action sequence A)
    ###########################################################

    # consolidate costs (ensemble_size x N, ) --> (N, )
    """Why don't they reshape costs"""
    new_costs = []
    for i in range(N):
        # 1-a0 1-a1 1-a2 ... 2-a0 2-a1 2-a2 ... 3-a0 3-a1 3-a2...
        new_costs.append(costs[i::N])  # start: stop: step
    
    

    # mean and std cost (across ensemble) [N,]
    mean_cost = np.mean(new_costs, 1)
    std_cost = np.std(new_costs, 1)

    if disc_logits is None: 
        # rank by rewards
        if evaluating:
            cost_for_ranking = mean_cost

        # sometimes rank by model disagreement, and sometimes rank by rewards
        else:
            if take_exploratory_actions:
                cost_for_ranking = mean_cost - 4 * std_cost
                print("   ****** taking exploratory actions for this rollout")
            else:
                cost_for_ranking = mean_cost
    else: 
        """
        new_costs.shape: [N, model_ens_size]
        disc_logits.shape: [horizon, model_ens_size, N]
        stepwise_costs.shape: []
        """
        # new_new_costs = np.reshape(costs, [-1, N]).T
        # assert np.allclose(new_new_costs, np.array(new_costs)), f"new_new_costs = {new_new_costs} != {new_costs} = new_costs"
        
        rewards = - np.reshape(stepwise_costs, [horizon, -1, N])
        # TODO: remove assertion
        assert np.allclose(-np.sum(rewards, axis = 0).T, np.array(new_costs))

        # transition_weights = softmax(np.sum(disc_logits, axis=0), axis=1)
        acc_transition_weights = softmax(np.cumsum(disc_logits, axis=0), axis=1)
        iw_rewards = np.sum(rewards * acc_transition_weights, axis = (0,1))
        iw_costs = - iw_rewards   
        cost_for_ranking = iw_costs

    return cost_for_ranking, mean_cost, std_cost
