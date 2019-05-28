import numpy as np
from scipy import optimize
from tabular import markov_chain, markov_decision_process, policies, gridworld


def reward_matrix_to_vector(mdp: markov_decision_process.MDP, reward_matrix: np.ndarray):
    return (mdp.T * reward_matrix).sum(axis=1)


def discounted_dual_linprog(mdp: markov_decision_process.MDP, reward: np.ndarray, discount_factor: float):
    transitions_sa_j = mdp.T
    transitions_s_a_j = transitions_sa_j.reshape((mdp.num_states, mdp.num_actions, mdp.num_states))
    constraint_matrix = (np.eye(mdp.num_states, mdp.num_states)[..., None] -
                         discount_factor * transitions_s_a_j.transpose(2, 0, 1))
    # noinspection PyTypeChecker
    decision_variables = optimize.linprog(
        c=-reward,
        A_eq=constraint_matrix.reshape((mdp.num_states, mdp.num_states * mdp.num_actions)),
        b_eq=np.ones((mdp.num_states,)),
        bounds=(0, np.inf),
        method='interior-point'
    ).x
    decision_variables = decision_variables.reshape(mdp.num_states, mdp.num_actions)
    return policies.Policy(decision_variables / decision_variables.sum(axis=1, keepdims=True))


def evaluate_discounted_q(mdp: markov_decision_process.MDP, policy: policies.Policy, reward: np.ndarray,
                          discount_factor: float) -> np.ndarray:

    return evaluate_discounted_chain_value(mdp.state_action_process(policy), reward, discount_factor)
    #return np.linalg.solve(np.eye(mdp.num_states * mdp.num_actions, mdp.num_states * mdp.num_actions) -
                           #discount_factor * (mdp.T @ policy.E()), reward)


def evaluate_discounted_v(mdp: markov_decision_process.MDP, policy: policies.Policy, reward: np.ndarray,
                          discount_factor: float) -> np.ndarray:
    return evaluate_discounted_chain_value(mdp.state_process(policy), policy.E() @ reward, discount_factor)


def evaluate_discounted_chain_value(chain: markov_chain.MarkovChain, reward: np.ndarray,
                                    discount_factor: float) -> np.ndarray:
    return np.linalg.solve(np.eye(chain.num_states, chain.num_states) - discount_factor * chain.T, reward)


def policy_iteration_discounted(mdp: markov_decision_process.MDP, reward: np.ndarray, discount_factor: float) -> \
        policies.Policy:
    policy = policies.Policy(np.ones((mdp.num_states, mdp.num_actions)) / mdp.num_actions)
    value = None
    while True:
        new_value = evaluate_discounted_q(mdp, policy, reward, discount_factor).reshape(
            (mdp.num_states, mdp.num_actions))
        if value is not None and np.all(np.allclose(new_value, value)):
            return policy

        value = new_value
        policy.policy[:] = 0
        policy.policy[np.arange(mdp.num_states), value.argmax(axis=1)] = 1


def value_iteration_discounted(mdp: markov_decision_process.MDP, reward: np.ndarray, discount_factor: float,
                               precision: float=1e-9):
    reward = reward.reshape((mdp.num_states, mdp.num_actions))
    value = np.max(reward, axis=1)
    max_diff = np.inf
    while max_diff > precision * (1 - discount_factor)/(2 * discount_factor):
        q_value = reward + discount_factor * (mdp.T @ value).reshape(mdp.num_states, mdp.num_actions)
        new_value = np.max(q_value, axis=1)

        max_diff = np.max(np.abs(new_value - value))
        value = new_value

    policy = np.zeros((mdp.num_states, mdp.num_actions))
    # noinspection PyUnboundLocalVariable
    policy[np.arange(mdp.num_states), np.argmax(q_value, axis=1)] = 1
    return policies.Policy(policy)


def solve_discounted(mdp: markov_decision_process.MDP, reward: np.ndarray, discount_factor: float):
    if mdp.num_states * mdp.num_actions > 500:
        return discounted_dual_linprog(mdp, reward, discount_factor)
    else:
        return policy_iteration_discounted(mdp, reward, discount_factor)


def _profile():
    from timeit import default_timer as timer

    discount_factor = 0.99

    for size in range(5, 30, 5):
        transition_grid = np.zeros((size, size))
        transition_grid[0, 0] = gridworld.S
        transition_grid[size - 1, size - 1] = gridworld.G
        reward_grid = np.zeros((size, size))
        reward_grid[size - 1, size - 1] = gridworld.G

        env = gridworld.Gridworld(transition_grid, reward_grid, 0.1)
        reward = reward_matrix_to_vector(env.mdp, env.reward_matrix)
        start = timer()
        for _ in range(20):
            value_iteration_discounted(env.mdp, reward, discount_factor)
        end = timer()
        print(f"VI({size}): {end - start}")
        start = timer()
        for _ in range(20):
            policy_iteration_discounted(env.mdp, reward, discount_factor)
        end = timer()
        print(f"PI({size}): {end - start}")
        start = timer()
        for _ in range(20):
            discounted_dual_linprog(env.mdp, reward, discount_factor)
        end = timer()
        print(f"IP({size}): {end - start}")
        start = timer()
        for _ in range(20):
            solve_discounted(env.mdp, reward, discount_factor)
        end = timer()
        print(f"solve({size}): {end - start}")


if __name__ == '__main__':
    _profile()
