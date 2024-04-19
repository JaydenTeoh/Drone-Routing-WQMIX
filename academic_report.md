## Methodology
To solve the drone routing problem, I decided to train decentralized but homogenous drone agent policies and a centralised action-value function using the WQMIX (Weighted QMIX) algorithm. WQMIX extends the QMIX algorithm by introducing weighting such that the action-value function can prioritize better joint actions. Although a centralized policy would have probably offered better benchmark performance, the rationale behind adopting decentralized policie swas to emulate real-world drone routing dynamics, where drones navigate autonomously based on local information.

Additionally, I leveraged RNNs to capture temporal dependencies and recurrent experience replay (from the [R2D2 paper](https://openreview.net/pdf?id=r1lyTjAqYX)) for prioritizing experiences with higher TD errors. Last but not least, for an extra boost to the benchmark performance of the trained models, I incorporated the benchmark environments within the training loop for few-shot learning.

## Takeaways
I'm grateful to the organizers for putting together this challenge; it has been a fantastic learning experience. As a junior researcher, experimenting with different MARL algorithms has deepened my understanding of the field.

Despite testing several SOTA algorithms like WQMIX and optimizing performance with experience replay and hyperparameter tuning, my algorithm was still unable to solve many of the benchmark environments. In hindsight, relying solely on fully decentralized policies may not have been the optimal strategy. I noticed there were instances of coordination failure due to agents being unaware of each other's goals. For example, in this [recording](assets/recording.mov), two of the agents rushed to their own goals and blocked the third agent from reaching its goal. 

After further research, I realised that this problem is well suited for using Message Passing Neural Networks (MPNN) which enable agents to communicate and coordinate in graph-based environments like this. I'm looking forward to trying out MPNNs for this problem after the competition.


## Replicating the results
The configurations I have used for training in each map is listed under the [configs](configs). Simply run the following command to train a model for a specific environment:

```
python train.py --map_name=<INCLUDE MAP NAME> --drone_num=<INCLUDE DRONE NUM>
```

Pretrained models are also available under [models](models/wqmix/). To test the pretrained models, adjust the `drone_num` and `map_name` variables in the main function of `policy_tester.py` accordingly and execute it.