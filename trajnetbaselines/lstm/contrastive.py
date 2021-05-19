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

        #for i in range(batch_split.shape[0] - 1):
        #    traj_primary = batch_scene[:, batch_split[i]] # [time, 2]
        #    traj_neighbor = batch_scene[:, batch_split[i]+1:batch_split[i+1]] # [time, num, 2]
        #    plot_scene(traj_primary, traj_neighbor, fname='scene_{:d}.png'.format(i))
        #import pdb; pdb.set_trace()

        # #####################################################
        #           TODO: fill the following code
        # #####################################################

        # -----------------------------------------------------
        #               Contrastive Sampling 
        # -----------------------------------------------------
        sample_pos, sample_neg = self._sampling_spatial(batch_scene, batch_split)
        
        # nan treatment
        nan_detect_neg=torch.isnan(sample_neg)
        sample_neg[nan_detect_neg]=0
        
        # -----------------------------------------------------
        #              Lower-dimensional Embedding 
        # -----------------------------------------------------
        
        emb_obsv = self.head_projection(batch_feat[:,batch_split[:-1],:])
        emb_pos = self.encoder_sample(sample_pos)
        emb_neg = self.encoder_sample(sample_neg)
        
        query = nn.functional.normalize(emb_obsv, dim=-1)
        key_pos = nn.functional.normalize(emb_pos, dim=-1)
        key_neg = nn.functional.normalize(emb_neg, dim=-1)
        
        # -----------------------------------------------------
        #                   Compute Similarity 
        # -----------------------------------------------------
        
        sim_pos = (query[0:self.horizon] * key_pos).sum(dim=-1)
        sim_neg = (query[0:self.horizon].unsqueeze(2) * key_neg.unsqueeze(1)).sum(dim=-1)
        
        sim_pos = sim_pos.reshape(sim_pos.shape[0]*sim_pos.shape[1],-1)
        sim_neg = sim_neg.reshape(sim_neg.shape[0]*sim_neg.shape[1],-1)
        
        logits = torch.cat([sim_pos, sim_neg], dim=-1) / self.temperature
        
        # -----------------------------------------------------
        #                       NCE Loss 
        # -----------------------------------------------------
        labels = torch.zeros(logits.size(0), dtype=torch.long)
        loss = self.criterion(logits, labels)
        
        return loss

    def event(self, batch_scene, batch_split, batch_feat):
        '''
            Social NCE with event samples, i.e., samples are spatial-temporal events at various time steps of the future
        '''
        raise ValueError("Optional")

    def _sampling_spatial(self, batch_scene, batch_split):

        gt_future = batch_scene[self.obs_length: self.obs_length+self.pred_length]

        # #####################################################
        #           TODO: fill the following code
        # #####################################################
        
        
        # -----------------------------------------------------
        #                  Positive Samples
        # -----------------------------------------------------
        gt_primary = gt_future[0:self.horizon, batch_split[0:batch_split.shape[0] - 1], :]
        sample_pos=gt_primary+torch.randn(gt_primary.size())*0.05

        # -----------------------------------------------------
        #                  Negative Samples
        # -----------------------------------------------------
        coll_zone = gt_future[0:self.horizon, batch_split[0]+1:batch_split[1]]
        for i in range(1,batch_split.shape[0] - 1):
            coll_zone=torch.cat((coll_zone, gt_future[0:self.horizon, batch_split[i]+1:batch_split[i+1]]),dim=1)

        sample_neg=coll_zone.unsqueeze(2)+self.agent_zone.unsqueeze(0).unsqueeze(0)
        sample_neg=sample_neg.reshape(sample_neg.size(0), sample_neg.size(1) * sample_neg.size(2), 2)

        sample_neg=sample_neg+torch.randn(sample_neg.size())*0.05
        
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
