<link href="assets\css\markdown.css"></link>

# Solving the [DRP Challenge](https://drp-challenge.com/#/overview) using Deep Reinforcement Learning

## Methodology
To solve the drone routing problem, I decided to train decentralized but homogenous drone agent policies and a centralised action-value function using the WQMIX (Weighted QMIX) algorithm. WQMIX extends the QMIX algorithm by introducing weighting such that the action-value function can prioritize better joint actions. Although a centralized policy would have probably offered better benchmark performance, the rationale behind adopting decentralized policies was to emulate real-world drone routing dynamics, where drones navigate autonomously based on local information.

Additionally, I leveraged RNNs to capture temporal dependencies and recurrent experience replay (from the [R2D2 paper](https://openreview.net/pdf?id=r1lyTjAqYX)) for prioritizing experiences with higher TD errors.

## Takeaways
I'm grateful to the organizers for putting together this challenge; it has been a fantastic learning experience. As a junior researcher, experimenting with different MARL algorithms has deepened my understanding of the field.

Despite testing several SOTA algorithms like WQMIX and optimizing performance with experience replay and hyperparameter tuning, my algorithm was still unable to solve many of the benchmark environments. In hindsight, relying solely on fully decentralized policies may not have been the optimal strategy. I noticed there were instances of coordination failure due to agents being unaware of each other's goals. For example, in this [recording](assets/recording.mov), two of the agents rushed to their own goals and blocked the third agent from reaching its goal. 

After further research, I realised that this problem is well suited for using Message Passing Neural Networks (MPNN) which enable agents to communicate and coordinate in graph-based environments like this. I'm looking forward to trying out MPNNs for this problem after the competition.


## Replicating the results
The configurations I have used for training in each map is listed under the [configs](configs). Simply run the following command to train a model for a specific environment:

```
git clone https://github.com/JaydenTeoh/Drone-Routing-WQMIX.git
pip3 install -e ./main
pip3 install -r ./main/requirements.txt
python train.py --map_name=<INCLUDE MAP NAME> --drone_num=<INCLUDE DRONE NUM>
```

Pretrained models are also available under [models](models/wqmix/). To test the pretrained models, adjust the `drone_num` and `map_name` variables in the main function of `policy_tester.py` accordingly and execute it.

The benchmark scores for the pretrained models can be found in the `benchmark_scores.json` file.

&nbsp;
# Original DRP Challenge Information

<a id="development"></a>

## Development  

<!-- #### ``policy/policy.py`` -->

In this competition, participants are expected to develop ``policy/policy.py``, which is essentially a mapping from input(``observation``) to output (``joint action``) at each step. 

- `observation (obs)`: The obs $s^i$ for each drone consists of two parts: `current location` and `goal position`.  They are in soft-hot representation: the length of this vector $s^i=\left[s_1^i, \ldots, s_j^i, \ldots s_{|V|}^i, s_{|V|+1}^i, \ldots, s_{|V|+j}^i, \ldots s_{|V|*2}^i\right]$ equates to the double of number $|V|$ of the nodes on a map.
  - It marks a node $s_j^i$ with 1 if the drone occupies it, while the rest remain zero.
  - For drones located on the edges, vector values are defined by: $s_j^i=1-\frac{len\left(l o c^i-v_j^i\right)}{len\left(v_j, v_k\right)}, s_k^i=1-s_j^i$ when drone $i$ traverses edge $\left(v_j, v_k\right)$, and 0 otherwise. Here, $loc^i=\left(l^{x^i}, l^{y^i}\right)$ represents drone $i$ 's current coordinates and len(,) represents the distance. As drone i approaches node $v_j^i$, the value of $s_j^i$ increases.
  - Also, it has Field of View information, which marks a node $s_j^i$ in onehot with -1 if another drone occupies it.
<p align="center">
 <img src="assets/img/obs.png" width="35%" >
  <img src="assets/img/framework.png" width="50%" >
</p>

- `joint action`: At each step, drones can choose a node to move. Consequently, we represent the action set $A$ using the node set $V$. It will wait at the current node if a drone choose an non-adjacent nodes. The joint action includes all individual actions from all drones. 
<!--
<p align="center">
 <img src="assets/img/policy.png" width="65%" >
</p>
-->

#### Step and Episode

Every time each drone takes action, increases step count.
In other words, every time the ``step`` function is excused, the number of steps increases by one.

The episode ends upon conflict, exceeding 100 steps, or all drones reaching goals and restarting with a new environment ( If not specified indications, only the positions of the start and goal change.).


#### Goal for Contribution
The goal for contribution in this competition is to minimize [cost](#cost) without collision happens.
You can test your developed (``policy/policy.py``) by loading it in ``policy_tester.py``.
> [!NOTE]
> Since drp is a gym-standard environment, you can develop it as an usual gym-standard environment without relying on ``policy_tester.py`` we provided. There is an [example code](example/) by using [pfrl](https://github.com/pfnet/pfrl).

<a id="evaluation"></a>

## Evaluation

We use three maps for evaluations: ``map_3x3``, ``map_aoba01``, ``map_shibuya``.

<p align="center">
  <img src="assets/img/map3_3.png" width="30%" >
  <img src="assets/img/map_aoba01.png" width="30%" >
  <img src="assets/img/map_shibuya.png" width="30%" >
</p>


Each map will be evaluated on various drone numbers and various start-goal pairs.
We call one pattern (fixed map, number of drones, and start-goal pair) as a problem and there are a totally of 30 problems which are defined in ``problem/problems.py``. (Participants are forbidden to alter this file.)

<a id="cost"></a>

#### Cost for each problem　

$$cost_p = \frac{1}{10} \sum_{i=1}^{10} \sum_{j \in drones} cost_{ij}
$$
Where:
$i$ is the iteration number. For each problem, we take average of 10 iterations as the final result.
$drones$ is the set of drones at that problem.

$$ cost_{ij} = \begin{cases}
step_{ij} & \text{if drone $j$ reached its goal without collision at iteration $i$, $cost_{ij}$ is the steps costed until reaching its goal } \\
100 & \text{if collision happened or drone $j$ doesn't reach the goal } \\
\end{cases}$$

There are three classic patterns to calculate costs as follows.
<p align="center">
<img src="assets/img/score_1.png" width="30%" >
<img src="assets/img/score_2.png" width="30%" >
<img src="assets/img/score_3.png" width="30%" >
</p>

#### Final cost of all problems
$$Final~Cost = \sum_{p \in problems}cost_p 
$$
where:
$cost_p$ is the cost of the problem $p$.

The final cost is the sum of the costs of the 30 problems ([more details](https://github.com/DrpChallenge/main/blob/main/problem/problems.py)). The objective is to **minimize** this final cost $Final~Cost$.

Once your (``policy/policy.py``) has been deployed, you can run ``calculate_cost.py``, which will outputs a json file (``your_team_name.json``) including the cost (named ``final cost``).


<a id ="appendix"></a>

## Appendix

Please refer to [this page](assets/markdown/appendix.md) to get more detailed information about the DRP environment. 
