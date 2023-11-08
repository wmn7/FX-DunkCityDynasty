import numpy as np
import torch as t
import torch.nn as nn
from ray.rllib.models.torch.recurrent_net import RecurrentNetwork
from ray.rllib.utils.annotations import override
from PlayerAgent.policy.GAT import GAT
from ray.rllib.models.modelv2 import ModelV2

class Network(RecurrentNetwork, nn.Module):

    player_typenum = 5
    tau = 1
    common_layers = nn.ParameterDict({
                    "type_embedding":None,
                    "embedding_act":None,
                    "RNN":None,
                    "GNN_0":None,
                    "GNN_1":None,
                    "action_nn":None,
                    "solo_value_nn":None,
                    "team_value_nn":None
                })   
    def __init__(self,
                 obs_space,
                 action_space,
                 num_outputs,
                 model_config,
                 name,
                 **kwargs):
        RecurrentNetwork.__init__(
            self, obs_space, action_space, None, model_config, name, **kwargs
        )
        nn.Module.__init__(self)
        """
        网络参数设置说明：
        MLP 神经网络输入大小：        102 - >  mlp_embedding_size: 512 (RNN input size)
        神经网络输出大小(GRU输入):    512 - >
        rnn_output_size 输出大小：  1024  
        GNN_0 input size: 1024  heading_size : 4
        GNN_0_output_size      :  1024
        GNN_1 input_size: 1024  heading_size : 4 
        GNN_1 output_size:   512
        action / value embeding_size : 512
        action output layer:  52
        value output layer:    1

        """
        # 找到网络参数
        model_config = {
            "mlp_embedding_size": int(512),
            "rnn_output_size": int(1024),
            "GNN_0_output_size": int(1024),
            "GNN_0_heading_size":int(4),
            "GNN_1_output_size": int(1024),
            "GNN_1_heading_size":int(4),
            "action_embeding_size": int(512),
            "value_embeding_size": int(512),
            "action_output_size": int(52),
            "value_output_size": int(1),
            "agent_num": int(6),
            "tau": 1.0,
            "alpha": 0.2
            
        }
        self.agent_obs_size = 102  # 72+30

        self.mlp_embedding_size = model_config["mlp_embedding_size"]
        self.rnn_output_size = model_config["rnn_output_size"]
        self.GNN_0_output_size = model_config["GNN_0_output_size"]
        self.GNN_0_heading_size = model_config["GNN_0_heading_size"]
        self.GNN_1_output_size = model_config["GNN_1_output_size"]
        self.GNN_1_heading_size = model_config["GNN_1_heading_size"]
        self.action_embeding_size = model_config["action_embeding_size"]
        self.value_embeding_size = model_config["value_embeding_size"]
        self.action_output_size = model_config["action_output_size"]
        self.value_output_size = model_config["value_output_size"]
        self.agent_num = model_config["agent_num"]
        self.tau = model_config["tau"]
        self.alpha = model_config["alpha"]

        if self.common_layers["type_embedding"] is None:
            # 设置5个embedding
            self.common_layers["type_embedding"] = nn.ModuleList()
            for _ in range(self.player_typenum):
                self.common_layers["type_embedding"].append(nn.Linear(self.agent_obs_size, self.mlp_embedding_size))
            self.common_layers["embedding_act"] = nn.ReLU()

            # 设置 RNN模块
            # batch first  [batch,seq_len,observation]
            
            self.common_layers["RNN"] = nn.GRU(self.mlp_embedding_size,
                                               self.rnn_output_size,
                                               1,
                                               batch_first=True)

            self.common_layers["GNN_0"] = GAT(self.rnn_output_size,
                                              self.GNN_0_output_size,
                                              self.GNN_0_heading_size,
                                              self.alpha,
                                              True)

            self.common_layers["GNN_1"] = GAT(self.GNN_0_output_size,
                                              self.GNN_1_output_size,
                                              self.GNN_1_heading_size,
                                              self.alpha,
                                              True)

            self.common_layers["action_nn"] = nn.ModuleList([
                nn.Sequential(nn.Linear(self.GNN_1_output_size, self.action_embeding_size),
                               nn.ReLU(),
                               nn.Linear(self.action_embeding_size, self.action_output_size))
                for _ in range(self.player_typenum)])

            self.common_layers["solo_value_nn"] = nn.ModuleList([
                nn.Sequential(nn.Linear(self.GNN_1_output_size, self.value_output_size),
                               nn.ReLU(),
                               nn.Linear(self.value_output_size, 1))
                for _ in range(self.player_typenum)
            ])

            self.common_layers["team_value_nn"] = nn.ModuleList([
                nn.Sequential(nn.Linear(self.GNN_1_output_size, self.value_output_size),
                               nn.ReLU(),
                               nn.Linear(self.value_output_size, 1)) for _ in range(self.player_typenum)
            ])

    @override(RecurrentNetwork)
    def forward(
            self,
            input_dict,
            state,
            seq_lens, ):
        assert seq_lens is not None
        # print("DEBUG input_dict_key", input_dict.keys())
        # torch.Size([32, 520])
        # TODO: self._wrapped_forward
        # wrapped_out, _ = self._wrapped_forward(input_dict, [], None)
        # input_dict["obs_flat"] = wrapped_out
        # input_dict["obs_flat"] 形状  [batch_size, vec_size]
        return super().forward(input_dict, state, seq_lens)

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
        global_vec_size = 30
        agent_num = 6
        action_mask_size = 52

        batch_size = inputs.shape[0]
        seq_len = inputs.shape[1]

        ## 输入形状处理
        # size: [batch_size, seq_Len, action_mask_size]
        action_masks = inputs[:, :, -action_mask_size:]
        # size: [batch_szie, seq_len, agent_num, global_size]
        global_inputs = inputs[:, :, :global_vec_size].unsqueeze(2).expand([-1, -1, agent_num, -1])
        # size: [batch_size, seq_len, agent_num, agent__vec_size]
        agent_inputs = inputs[:, :, global_vec_size:-action_mask_size].reshape(batch_size, seq_len, agent_num, -1)

        # size:  [batch_size,seq_len,agent_num,agent_vec_size+global_vec_size]
        batch_seq_agent_inputs = t.cat([agent_inputs, global_inputs], dim=-1)
        # size:  [batch_size,agent_num,seq_len,agent_vec_size+global_vec_size]
        batch_agent_seq_inputs = batch_seq_agent_inputs.transpose(1, 2)
        # flatten: [batch_size*agent_num,seq_len,agent_vec_size+global_vec_size]
        batch_agent_flat_seq_inputs = batch_agent_seq_inputs.reshape(batch_size * agent_num, seq_len, -1)

        # hidden_state 处理
        # print("debug", len(state))
        # print("debug", state[0].shape)
        batch_agent_hidden_state = state[0].reshape(batch_size * agent_num, -1)

        # 获取每一个智能体的player type:  在此假设每一个agent的observation 第一个index表示player_type
        # 对其进行embedding
        GNN_inputs_list = []
        output_hidden_states_list = []

        # batch_size*agent_num
        player_type_list = []

        # 经过RNN，得到 GNN embedding
        for i in range(batch_size * agent_num):
            # size: [seq_len,agent_vec_size+global_vec_size]
            vec = batch_agent_flat_seq_inputs[i, :, 1:]
            player_type_index = int(vec.cpu().numpy()[0, 0])
            embedding = self.common_layers["embedding_act"](
                self.common_layers["type_embedding"][player_type_index](vec)
            )
            # RNN 操作
            print('MLP embedding', embedding.shape) # torch.Size([1, 512])
            # print('batch_agent_hidden_state_i', batch_agent_hidden_state.shape) # torch.Size([1, 1024]
            rnn_embedding, h = self.common_layers["RNN"](embedding, batch_agent_hidden_state[i].unsqueeze(0))
            print('rnn_embedding', rnn_embedding.shape) # torch.Size([1, 1024])
            # rnn embedding [seq_len, size]  -> [size]
            GNN_inputs_list.append(rnn_embedding[-1,:])
            output_hidden_states_list.append(h)

            player_type_list.append(player_type_index)

        player_type_np = np.array(player_type_list)
        # size: [batch_size,agent_num]
        player_type_np = player_type_np.reshape(-1, agent_num)
        # size: [batch_size]
        self.player_type_np = player_type_np[:, 0]

        # [batch,agent_num, -1]
        GNN_input = t.cat(GNN_inputs_list, dim=0).reshape(batch_size, agent_num, -1)
        # [batch*agent_num, -1]
        output_hidden_states = t.cat(output_hidden_states_list, dim=0)
        # 进入GNN
        # size: [batch_size,agent_num, -1]
        GNN_0 = self.common_layers["GNN_0"](GNN_input)
        GNN_1 = self.common_layers["GNN_1"](GNN_0)
        # size: [batch_size,agent_num, -1]
        self._features = GNN_1
        print("self._features", self._features.shape) # torch.Size([32, 6, 1024])

        # 经过最终的action policy + value policy

        # 经过 action_policy
        # size:  [batch_size,output_size]
        action_logit = t.cat([self.common_layers["action_nn"][self.player_type_np [i]](self._features[i, 0, :]).unsqueeze(0) for i in range(batch_size)],
                             dim=0)
        action_logit = action_logit.unsqueeze(1)
        # value :
        # solo_value = t.cat([ self.common_layers["solo_value_nn"][i](self._features[i,0,:]) for i in range(batch_size)],dim=0)
        # team_value = t.cat([ self.common_layers["team_value_nn"][i](self._features[i,0,:]+self._features[i,2,:]+self._features[i,2,:])
        #                     for i in range(batch_size)],dim=0)
        
        # 加入 action mask
        zero_matrix = t.zeros_like(action_logit)
        action_logit = t.where(action_masks != 0, action_logit, zero_matrix)
        print("action_logit", action_logit.shape) # torch.Size([32, 1, 52]

        return action_logit, [output_hidden_states]

    def value_function(self):
        """
        @override(ModelV2)
        def value_function(self):
            assert self._cur_value is not None, "must call forward() first"
            return self._cur_value

        """
        assert self._features is not None, "must call forward() first"
        batch_size = self._features.shape[0]
        solo_value = t.cat([self.common_layers["solo_value_nn"][self.player_type_np[i]](self._features[i, 0, :]) for i in range(batch_size)],
                           dim=0)
        team_value = t.cat([self.common_layers["team_value_nn"][self.player_type_np[i]](
            self._features[i, 0, :] + self._features[i, 2, :] + self._features[i, 2, :])
            for i in range(batch_size)], dim=0)
        value = self.tau * solo_value + (1 - self.tau) * team_value
        return t.reshape(value, [-1])

    @override(ModelV2)
    def get_initial_state(self):
        """
        @override(ModelV2)
        def get_initial_state(self):
            # Place hidden states on same device as model.
            h = [self.fc1.weight.new(
                1, self.rnn_hidden_dim).zero_().squeeze(0)]
            return h
        """
        return [t.zeros((1, self.rnn_output_size*self.agent_num))]
