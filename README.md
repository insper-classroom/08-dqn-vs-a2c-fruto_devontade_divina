### A2C CartPole

To understand better the A2C algorithm and the best hyperparameters for it, the cartpole environment was loaded with an agent using said algorithm and the best set of parameters for it were:

```python
model_a2c = A2C(
    policy="MlpPolicy",
    env=env,
    learning_rate=0.0007,  
    n_steps=10,            
    gamma=0.99,            
    gae_lambda=1.0,      
    ent_coef=0.0,         
    vf_coef=0.5,           
    max_grad_norm=0.5,     
    device='cpu'
)
```
And its learning curve can be seen below:
<p align="center">
  <img src="images/a2c_cartpole1.png" width="1000">  <br>
</p>


with those parameters the agent successfully learned to solve the environment with an average evaluation reward of:

```python
Mean reward: 500.0 +/- 0.00
```

which was the best amongst agents trained with default and other parameters. Also, forcing the agent to run on the cpu greatly improved the execution time, as it seems a2c cannot leverage the gpu's capabilitys efficiently.

### A2C vs DQN CartPole

A stable baseline DQN based agent was loaded with the ideal parameters for DQN found in the previous assignment. Even though the A2C agent managed to successfully solve the environment, again, the DQN agent proved unsuccessfull.

<p align="center">
  <img src="images/a2c_vs_dqn.png" width="1000">  <br>
</p>



