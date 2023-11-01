from rllib.models.torch.recurrent_net import RecurrentNetwork
import torch.nn as nn
import torch as t
import numpy as np
from GAT import GraphAttentionLayer

class Network(RecurrentNetwork, nn.Module):
    common_layers = {
        "type_bedding": None,
        "embeding_act": None,
        "RNN": None,
        "GNN_0": None,
        "GNN_1":None,
        "action_nn":None,
        "solo_value_nn":None,
        "team_value_nn":None
    }
    player_typenum = 5
    tau = 1

    def __init__(self,
        obs_space,
        action_space,
        num_outputs,
        model_config,
        name):
        nn.Module.__init__(self)
        super(Network, self).__init__(
            obs_space, action_space, None, model_config, name)
        
        """
        网络参数设置说明：
        MLP 神经网络输入大小：  102 - >  
        MLP 神经网络输出大小(GRU输入)：  512 - >
        GRU 输出大小：        1024
        embedding 输入：      256,  头数: 4
        GNN_0 输出：          256 * 4
        GNN_1 输入:           256  头数：4
        GNN_1 输出：          128 * 4
        action / value 输入 : 512
        action hidden layers + output layer:  [256, 52]
        value hidden layer + output layer:    [256, 1]
        """

        # 找到网络参数 
        # TODO : CHECK
        self.agent_obs_size = 102   # 72+30
        self.embedding_size = model_config["embedding_size"]
        self.action_dim = 52
        self.num_outputs = num_outputs
        self.GNN_output_size = model_config["GNN_output_size"]
        self.GNN_heading_size = model_config["GNN_heading_size"]
        
        self.action_policy_size = model_config["action_policy_size"]
        self.value_policy_size = model_config["value_policy_size"]

        self.agent_num = model_config["agent_num"]

        self.tau = model_config["tau"]

        # 设置全局网络
        if self.common_layers["type_bedding"] is None:
            # 设置5个embedding
            self.common_layers["type_bedding"] = nn.ModuleList()
            for _ in range(self.player_typenum):
                self.common_layers["type_bedding"].append(nn.Linear(self.agent_obs_size,self.embedding_size))
            self.common_layers["embeding_act"] = nn.functional.relu

            # 设置 RNN模块
            # batch first  [batch,seq_len,observation]
            self.common_layers["RNN"] = nn.GRU(self.embedding_size,
                                                   self.num_outputs,
                                                   1,
                                                   batch_first = True)
            
            self.common_layers["GNN_0"] = nn.ModuleList([GraphAttentionLayer(self.embedding_size//self.GNN_heading_size,
                                                                             self.GNN_heading_size,
                                                                             dropout=False,
                                                                             alpha=0,
                                                                             concat=True) for _ in range(self.GNN_heading_size)])
            
            self.common_layers["GNN_1"] = nn.ModuleList([GraphAttentionLayer(self.GNN_heading_size//self.GNN_heading_size,
                                                                             self.GNN_heading_size,
                                                                             dropout=False,
                                                                             alpha=0,
                                                                             concat=True) for _ in range(self.GNN_heading_size)])
        
            self.common_layers["action_nn"] = nn.ModuleList([
                nn.Sequential([nn.Linear(self.GNN_output_size,self.action_policy_size),
                               nn.functional.relu,
                               nn.Linear(self.action_policy_size,self.action_dim)]) for _ in range(self.player_typenum)
            ])

            self.common_layers["solo_value_nn"] = nn.ModuleList([
                nn.Sequential([nn.Linear(self.GNN_output_size,self.value_policy_size),
                               nn.functional.relu,
                               nn.Linear(self.value_policy_size,1)]) for _ in range(self.player_typenum)
            ])

            self.common_layers["team_value_nn"] = nn.ModuleList([
                nn.Sequential([nn.Linear(self.GNN_output_size,self.value_policy_size),
                               nn.functional.relu,
                               nn.Linear(self.value_policy_size,1)] )for _ in range(self.player_typenum)
            ])

    def forward(
        self,
        input_dict,
        state,
        seq_lens,):
        assert seq_lens is not None
        wrapped_out, _ = self._wrapped_forward(input_dict, [], None)
        input_dict["obs_flat"] = wrapped_out
        # input_dict["obs_flat"] 形状  [batch_size, vec_size]
        return super().forward(input_dict["obs_flat"], state, seq_lens)
        

    def forward_rnn(self, inputs, state, seq_lens):
        # 输入
        # observation size: 520
        # global state: 30
        # slel_state / ally / enemy :  73
        # action mask: 52

        # inputs: [batch_size, seq_len, observation]
        # states: [batch_size, observation]
        # size
        observation_vec_size = 520
        global_vec_size =  30
        agent_num = 6
        action_mask_size = 52

        batch_size = inputs.shape[0]
        seq_len = inputs.shape[1]

        ## 输入形状处理
        # size: [batch_size, seq_Len, action_mask_size]
        action_masks = inputs[:,:,-action_mask_size:] 
        # size: [batch_szie, seq_len, agent_num, global_size]
        global_inputs = inputs[:,:,:global_vec_size].unsqueeze(2).expand([-1,-1,agent_num,-1])
        # size: [batch_size, seq_len, agent_num, agent__vec_size]
        agent_inputs = inputs[:,:,global_vec_size:-action_mask_size].reshape(batch_size,seq_len,agent_num,-1)
        
        # size:  [batch_size,seq_len,agent_num,agent_vec_size+global_vec_size]
        batch_seq_agent_inputs = t.cat([agent_inputs,global_inputs],dim=-1)
        # size:  [batch_size,agent_num,seq_len,agent_vec_size+global_vec_size]
        batch_agent_seq_inputs = batch_seq_agent_inputs.transpose(1,2)
        # flatten: [batch_size*agent_num,seq_len,agent_vec_size+global_vec_size]
        batch_agent_flat_seq_inputs = batch_agent_seq_inputs.view(batch_size*agent_num,seq_len,-1)

        # hidden_state 处理
        batch_agent_hidden_state = state.view(batch_size*agent_num,-1)

        # 获取每一个智能体的player type:  在此假设每一个agent的observation 第一个index表示player_type
        # 对其进行embedding
        GNN_inputs_list = []
        output_hidden_states_list = []
        
        # batch_size*agent_num
        player_type_list = []

        # 经过RNN，得到 GNN embedding
        for i in range(batch_size*agent_num):
            # size: [seq_len,agent_vec_size+global_vec_size]
            vec = batch_agent_flat_seq_inputs[i,:,1:]
            player_type_index = vec.cpu().numpy()[0,0]
            embedding = self.common_layers["embeding_act"](
                self.common_layers["type_bedding"][player_type_index](vec)
            )

            # RNN 操作
            rnn_embedding, h = self.common_layers["RNN"](embedding,batch_agent_hidden_state[i]) 
            
            GNN_inputs_list.append(rnn_embedding)
            output_hidden_states_list.append(h)

            player_type_list.append(player_type_index)

        player_type_np = np.array(player_type_list)
        # size: [batch_size,agent_num]
        player_type_np = player_type_np.reshape(-1,agent_num)
        # size: [batch_size]
        player_type_np = player_type_np[:,0]

        # [batch,agent_num, -1]
        GNN_input = t.cat(GNN_inputs_list,dim=0).view(batch_size,agent_num,-1)
        # [batch*agent_num, -1]
        output_hidden_states = t.cat(output_hidden_states_list,dim=0)
        # 进入GNN
        # size: [batch_size,agent_num,-1]
        GNN_0 = t.cat([t.cat([gat(GNN_input[i,:,:]) for gat in self.common_layers["GNN_0"]],-1) for i in range(batch_size)])

        # size: [batch_size,agent_num,-1]
        self._features = [t.cat([ gat(GNN_0[i,:,:]) for gat in self.common_layers["GNN_1"] ]) for i in range(batch_size)]

        # 经过最终的action policy + value policy
        
        # 经过 action_policy
        # size:  [batch_size,output_size]
        action_logit = t.cat([ self.common_layers["action_nn"][i](self._features[i,0,:]) for i in range(batch_size)],dim=0)
        action_logit = action_logit.unsqueeze(1)
        # value : 
        # solo_value = t.cat([ self.common_layers["solo_value_nn"][i](self._features[i,0,:]) for i in range(batch_size)],dim=0)
        # team_value = t.cat([ self.common_layers["team_value_nn"][i](self._features[i,0,:]+self._features[i,2,:]+self._features[i,2,:]) 
        #                     for i in range(batch_size)],dim=0)
        
        # 加入 action mask
        zero_matrix = t.zeros_like(action_logit)
        action_logit = t.where(action_masks!=0,action_logit,zero_matrix)

        return action_logit, output_hidden_states
    
    def value_function(self):
        assert self._features is not None, "must call forward() first"
        batch_size = self._features.shape[0]
        solo_value = t.cat([ self.common_layers["solo_value_nn"][i](self._features[i,0,:]) for i in range(batch_size)],dim=0)
        team_value = t.cat([ self.common_layers["team_value_nn"][i](self._features[i,0,:]+self._features[i,2,:]+self._features[i,2,:]) 
                            for i in range(batch_size)],dim=0)
        value = self.tau*solo_value + (1-self.tau)*team_value
        return t.reshape(value,[-1])
        
    def  get_initial_state(self):
        return t.zeros((1,self.num_outputs))