import gym
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ''))
from policy.policy_wqmix import policy

def policy_evaluation(policy, drone_num, map_name, reward_list, start, goal, render):
    if not start or goal:
        assert drone_num == len(start) and drone_num == len(
            goal
        ), "The number of elements in start and goal list does not match with drone_num."
        assert not any(
            element in start for element in goal
        ), "The elements of goal and start must not match."
    print("drp_env:drp-" + str(drone_num) + "agent_" + map_name + "-v2")
    env = gym.make(
        "drp_env:drp-" + str(drone_num) + "agent_" + map_name + "-v2",
        state_repre_flag="onehot_fov",
        reward_list=reward_list,
        goal_array=goal,
        start_ori_array=start,
    )
    obs = env.reset()
    print(f"observation_space:{env.observation_space}")
    print(f"action_space:{env.action_space}")

    done_all = False
    rnn_hidden_state = None
    while not done_all:
        if render == True:
            env.render()
        print(f"obs:{obs}")  # current global observation

        """
        INPUT: takes in current obs, environment and the previous rnn_hidden_state
        OUTPUT: returns action and new rnn_hidden_state
        """
        actions, rnn_hidden_state = policy(obs, env, rnn_hidden_state)
        obs, reward, done, info = env.step(
            actions
        )  # transfer to next state once joint action is taken
        print(f"obs:{obs}, actions:{actions}, reward:{reward}, done:{done},info:{info}")
        done_all = all(done)
        env.get_obs()


if __name__ == "__main__":
    drone_num = 4  # the number of drones (min:2 max:30)
    map_name = "map_aoba01"  # the map name (available maps: "map_3x3","map_aoba01","map_osaka" )

    # reward_list is individual reward function where
    # "goal: 100" means one drone will obtain 100 rewards once it reach its goal.
    # Similarly, "collision"/"wait"/"move" are rewards when a collision happens/one drone wait one step/moves one step;
    reward_list = {
        "goal": 100,
        "collision": -10,
        "wait": -10,
        "move": -1,
    }  # Developers can freely to alter the reward function (rewards are not used as evaluation index)

    # If the start and goal are empty lists, they are randomly selected.
    start = [11, 9, 2, 7] # drone1's start: node 0;  drone2's start: node 2;  drone3's start: node 4;
    goal = [8, 12, 15, 10]  # drone1's goal: node 3;  drone2's goal: node 6;  drone3's goal: node 1;
    render = True  # Choose whether to visualize

    """
    policy_evaluation() function is used to evaluate the "policy" developed by participants
    participants are expected to develop "policy",
    which is essentially a mapping from input(global observation) to output(joint action) at each step
    """
    policy_evaluation(
        policy=policy,  # this is an example policy
        drone_num=drone_num,
        map_name=map_name,
        reward_list=reward_list,
        goal=goal,
        start=start,
        render=render,
    )
