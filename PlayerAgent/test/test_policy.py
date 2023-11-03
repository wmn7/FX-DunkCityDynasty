import torch
from PlayerAgent.policy.policy import Network
from gymnasium import spaces

# inputs: [batch_size, seq_len, observation]  -- both batch_size and seq_len are changing everytime
# states[0]: [batch_size, observation]  -- batch_size is changing everytime

if __name__=="__main__":

    inputs = torch.load('./inputs.pt')
    state_0 = torch.load('./state.pt')

    print("inputs.shape: ", inputs.shape)
    print("state[0].shape: ", state_0.shape)
    # inputs.shape:  torch.Size([4, 8, 520])
    # state_0.shape:  torch.Size([4, 1, 1536])

    obs_space = spaces.Discrete(100)
    action_space = spaces.Discrete(52)

    network = Network(obs_space,action_space,None,{},"custom_policy")
    state_0 = [torch.cat([network.get_initial_state()[0] for _ in range(4)],dim=0)]
    print("============== into the network forward =========")
    # network.forward_rnn(inputs[:,0,:].unsqueeze(1),state_0,None)
    network.forward_rnn(inputs,state_0,None)
    print("============== out of the network ================")
    print("============== into the value function ===========")
    network.get_initial_state()
    print("============== out of the value function =========")

