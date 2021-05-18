import math
import torch
import torch.nn as nn

class SocialNCE():
    '''
        Social NCE: Contrastive Learning of Socially-aware Motion Representations (https://arxiv.org/abs/2012.11717)
    '''
    def __init__(self, obs_length, pred_length, head_projection, encoder_sample, temperature, horizon, sampling):

        # problem setting
        self.obs_length = obs_length
        self.pred_length = pred_length

        # nce models
        self.head_projection = head_projection
        self.encoder_sample = encoder_sample

        # nce loss
        self.criterion = nn.CrossEntropyLoss()

        # nce param
        self.temperature = temperature
        self.horizon = horizon

        # sampling param
        self.noise_local = 0.1
        self.min_seperation = 0.2
        self.agent_zone = self.min_seperation * torch.tensor([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0], [0.707, 0.707], [0.707, -0.707], [-0.707, 0.707], [-0.707, -0.707], [0.0, 0.0]])

    def spatial(self, batch_scene, batch_split, batch_feat):
        '''
            Social NCE with spatial samples, i.e., samples are locations at a specific time of the future
            Input:
                batch_scene: coordinates of agents in the scene, tensor of shape [obs_length + pred_length, total num of agents in the batch, 2]
                batch_split: index of scene split in the batch, tensor of shape [batch_size + 1]
                batch_feat: encoded features of observations, tensor of shape [pred_length, scene, feat_dim]
            Output:
                loss: social nce loss
        '''

        # -----------------------------------------------------
        #               Visualize Trajectories 
        #       (Use this block to visualize the raw data)
        # -----------------------------------------------------

        for i in range(batch_split.shape[0] - 1):
            traj_primary = batch_scene[:, batch_split[i]] # [time, 2]
            traj_neighbor = batch_scene[:, batch_split[i]+1:batch_split[i+1]] # [time, num, 2]
            plot_scene(traj_primary, traj_neighbor, fname='scene_{:d}.png'.format(i))
        print(self.noise_local)
        import pdb; pdb.set_trace()

        # #####################################################
        #           TODO: fill the following code
        # #####################################################

        # -----------------------------------------------------
        #               Contrastive Sampling 
        # -----------------------------------------------------
        (sample_pos, sample_neg) = self._sampling_spatial(batch_scene, batch_split)
        # -----------------------------------------------------
        #              Lower-dimensional Embedding 
        # -----------------------------------------------------
        
        # interestsID = batch_split[0:-1] #to erase
        
        # ------------------------ to erase ------------------
        # embedding
        #emb_obsv = self.head_projection(feat[:, :, :1])
        #emb_pos = self.encoder_sample(candidate_pos, time_pos[:, :, None])
        #emb_neg = self.encoder_sample(candidate_neg, time_neg[:, :, :, None])
        # ------------------------ to erase end ------------------
        emb_obsv = self.head_projection(batch_feat[self.obs_length,batch_split,:]) #comprendre come ça marche le head_projection et encoder_sample
        emb_pos = self.encoder_sample(sample_pos)
        emb_neg = self.encoder_sample(torch.tensor(sample_neg))
        
        # ------------------------ to erase ------------------
        # normalization
        #query = nn.functional.normalize(emb_obsv, dim=-1)
        #key_pos = nn.functional.normalize(emb_pos, dim=-1)
        #key_neg = nn.functional.normalize(emb_neg, dim=-1)
        # ------------------------ to erase end ------------------
        query = nn.functional.normalize(emb_obsv, dim=-1)
        key_pos = nn.functional.normalize(emb_pos, dim=-1)
        key_neg = nn.functional.normalize(emb_neg, dim=-1)
        
        # -----------------------------------------------------
        #                   Compute Similarity 
        # -----------------------------------------------------
        
        # similarity
        #sim_pos = (query * key_pos.unsqueeze(1)).sum(dim=-1)
        #sim_neg = (query.unsqueeze(2) * key_neg.unsqueeze(1)).sum(dim=-1)
        sim_pos = (query[:,None,:] * key_pos.unsqueeze[ :,None,:]).sum(dim=-1) #pdf 3.1 -> que est ce que sont les dimension de query, key_neg, key_pos
        sim_neg = (query.unsqueeze[:,None,:] * key_neg).sum(dim=-1)
        
        # !!! gerer les nan !!!
        # !!! start from LSTM trained last time !!!
        
        # logits
        #sim_pos_avg = sim_pos.mean(axis=1)              # average over samples
        #sz_neg = sim_neg.size()
        #sim_neg_flat = sim_neg.view(sz_neg[0], sz_neg[1]*sz_neg[2], sz_neg[3])
        #logits = torch.cat([sim_pos_avg.view(sz_neg[0]*sz_neg[3], 1), sim_neg_flat.view(
        #    sz_neg[0]*sz_neg[3], sz_neg[1]*sz_neg[2])], dim=1) / self.temperature
        logits = torch.cat([sim_pos, sim_neg], dim=-1) / self.temperature

        
        # -----------------------------------------------------
        #                       NCE Loss 
        # -----------------------------------------------------
        
        # loss
        #labels = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)
        #loss = self.criterion(logits, labels)
        labels = torch.zeros(logits.size(0), dtype=torch.long)
        loss = self.criterion(logits, labels)
        print('the contrast loss is: ', loss)
        
        return loss

    def event(self, batch_scene, batch_split, batch_feat):
        '''
            Social NCE with event samples, i.e., samples are spatial-temporal events at various time steps of the future
        '''
        raise ValueError("Optional")

    def _sampling_spatial(self, batch_scene, batch_split, min_dist_ba, c_eps):

        gt_future = batch_scene[self.obs_length: self.obs_length+self.pred_length]

        # #####################################################
        #           TODO: fill the following code
        # #####################################################
        
        gt_primary = gt_future[:, batch_split[i]] # [pred_length, 2]
        gt_neighbor = gt_future[:, batch_split[i]+1:batch_split[i+1]] # [pred_length, num, 2]

        # -----------------------------------------------------
        #                  Positive Samples
        # -----------------------------------------------------
        
        epsilon_pos = torch.rand(gt_primary.size())*c_eps
        sample_pos = gt_primary + epsilon_pos

        # -----------------------------------------------------
        #                  Negative Samples
        # -----------------------------------------------------
        
        epsilon_neg = torch.rand(gt_neighbor.size())*c_eps
       # theta =
    # delta_s = torch.ones((gt_neighbor.size(0),gt_neighbor.size(1)*8,gt_neighbor.size(2)))*min_dist_ba
        
        sample_pos = gt_neighbor + 10

        # -----------------------------------------------------
        #       Remove negatives that are too hard (optional)
        # -----------------------------------------------------
        
        # -----------------------------------------------------
        #       Remove negatives that are too easy (optional)
        # -----------------------------------------------------

        return sample_pos, sample_neg

class EventEncoder(nn.Module):
    '''
        Event encoder that maps an sampled event (location & time) to the embedding space
    '''
    def __init__(self, hidden_dim, head_dim):

        super(EventEncoder, self).__init__()
        self.temporal = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(inplace=True)
            )
        self.spatial = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.ReLU(inplace=True)
            )
        self.encoder = nn.Sequential(
            nn.Linear(hidden_dim*2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, head_dim)
        )

    def forward(self, state, time):
        emb_state = self.spatial(state)
        emb_time = self.temporal(time)
        out = self.encoder(torch.cat([emb_time, emb_state], axis=-1))
        return out

class SpatialEncoder(nn.Module):
    '''
        Spatial encoder that maps an sampled location to the embedding space
    '''
    def __init__(self, hidden_dim, head_dim):
        super(SpatialEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, head_dim)
        )

    def forward(self, state):
        return self.encoder(state)

class ProjHead(nn.Module):
    '''
        Nonlinear projection head that maps the extracted motion features to the embedding space
    '''
    def __init__(self, feat_dim, hidden_dim, head_dim):
        super(ProjHead, self).__init__()
        self.head = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, head_dim)
            )

    def forward(self, feat):
        return self.head(feat)

def plot_scene(primary, neighbor, fname):
    '''
        Plot raw trajectories
    '''
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig = plt.figure(frameon=False)
    fig.set_size_inches(16, 9)
    ax = fig.add_subplot(1, 1, 1)

    ax.plot(primary[:, 0], primary[:, 1], 'k-')
    for i in range(neighbor.size(1)):
        ax.plot(neighbor[:, i, 0], neighbor[:, i, 1], 'b-.')

    ax.set_aspect('equal')
    plt.grid()
    plt.savefig(fname, bbox_inches='tight', pad_inches=0)
    plt.close(fig)
