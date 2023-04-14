import torch
import torch.nn as nn
import torch.nn.functional as F
from model import common

# Global Learnable Attention
class GLA(nn.Module):

    def __init__(self, n_hashes=4, channels=64, k_size=3, reduction=4, chunk_size=144, conv=common.default_conv, res_scale=1):
        super(GLA,self).__init__()
        self.chunk_size = chunk_size
        self.n_hashes = n_hashes
        self.reduction = reduction
        self.res_scale = res_scale
        self.conv_match = common.BasicBlock(conv, channels, channels//reduction, k_size, bn=False, act=nn.ReLU(inplace=True))
        self.conv_assembly = common.BasicBlock(conv, channels, channels, k_size, bn=False, act=nn.ReLU(inplace=True))
        self.conv_assembly_fc = common.BasicBlock(conv, channels, channels, k_size, bn=False, act=nn.ReLU(inplace=True))
        self.fc = nn.Sequential(
            nn.Linear(channels, chunk_size),
            nn.ReLU(inplace=True),
            nn.Linear(chunk_size, chunk_size)
        )

    # Super-Bit Locality-Sensitive Hashing
    def SBLSH(self, hash_buckets, x):
        #x: [N,H*W,C]
        N = x.shape[0]
        device = x.device

        #generate random rotation matrix
        rotations_shape = (1, x.shape[-1], self.n_hashes, hash_buckets//2) #[1,C,n_hashes,hash_buckets//2]
        # assert rotations_shape[1] > rotations_shape[2]*rotations_shape[3]
        random_rotations = torch.nn.init.orthogonal_(torch.empty(x.shape[-1], hash_buckets))
        for _ in range(self.n_hashes-1):
            random_rotations = torch.cat([random_rotations, torch.nn.init.orthogonal_(torch.empty(x.shape[-1],hash_buckets))], dim=-1)
        # Training under multi-gpu: random_rotations.cuda() -> random_rotations.to(x.device) (suggested by Breeze-Zero from github: https://github.com/laoyangui/DLSN/issues/2)
        random_rotations = random_rotations.reshape(rotations_shape[0], rotations_shape[1], rotations_shape[2], hash_buckets).expand(N, -1, -1, -1).cuda() #[N, C, n_hashes, hash_buckets]
        rotated_vecs = torch.einsum('btf,bfhi->bhti', x, random_rotations) #[N, n_hashes, H*W, hash_buckets]

        #get hash codes
        hash_codes = torch.argmax(rotated_vecs, dim=-1) #[N,n_hashes,H*W]

        #add offsets to avoid hash codes overlapping between hash rounds
        offsets = torch.arange(self.n_hashes, device=device)
        offsets = torch.reshape(offsets * hash_buckets, (1, -1, 1))
        hash_codes = torch.reshape(hash_codes + offsets, (N, -1,)) #[N,n_hashes*H*W]

        return hash_codes

    def add_adjacent_buckets(self, x):
        x_extra_back = torch.cat([x[:,:,-1:, ...], x[:,:,:-1, ...]], dim=2)
        x_extra_forward = torch.cat([x[:,:,1:, ...], x[:,:,:1,...]], dim=2)
        return torch.cat([x, x_extra_back,x_extra_forward], dim=3)

    def forward(self, input):

        N,_,H,W = input.shape
        x_embed = self.conv_match(input).view(N,-1,H*W).contiguous().permute(0,2,1)
        y_embed = self.conv_assembly(input).view(N,-1,H*W).contiguous().permute(0,2,1)
        fc_embed = self.conv_assembly_fc(input).view(N,-1,H*W).contiguous().permute(0,2,1)
        x_embed_extra_index = torch.arange(H * W).unsqueeze(0).unsqueeze(0).permute(0, 2, 1).cuda() # [1, HW, 1]

        L,C = x_embed.shape[-2:]

        #number of hash buckets/hash bits
        hash_buckets = min(L//self.chunk_size + (L//self.chunk_size)%2, 128)

        #get assigned hash codes/bucket number
        hash_codes = self.SBLSH(hash_buckets, x_embed) #[N,n_hashes*H*W]
        hash_codes = hash_codes.detach()

        #group elements with same hash code by sorting
        _, indices = hash_codes.sort(dim=-1) #[N,n_hashes*H*W]
        _, undo_sort = indices.sort(dim=-1) #undo_sort to recover original order
        mod_indices = (indices % L) #now range from (0->H*W)

        x_embed_sorted = common.batched_index_select(x_embed, mod_indices) #[N,n_hashes*H*W,C]
        y_embed_sorted = common.batched_index_select(y_embed, mod_indices) #[N,n_hashes*H*W,C]
        fc_embed_embed_sorted = common.batched_index_select(fc_embed, mod_indices) #[N,n_hashes*H*W,C]

        #pad the embedding if it cannot be divided by chunk_size
        padding = self.chunk_size - L%self.chunk_size if L%self.chunk_size!=0 else 0
        x_att_buckets = torch.reshape(x_embed_sorted, (N, self.n_hashes,-1, C)) #[N, n_hashes, H*W,C]
        y_att_buckets = torch.reshape(y_embed_sorted, (N, self.n_hashes,-1, C*self.reduction))
        fc_att_buckets = torch.reshape(fc_embed_embed_sorted, (N, self.n_hashes,-1, C*self.reduction))

        if padding:
            pad_x = x_att_buckets[:,:,-padding:,:].clone()
            pad_y = y_att_buckets[:,:,-padding:,:].clone()
            pad_fc = fc_att_buckets[:,:,-padding:,:].clone()
            x_att_buckets = torch.cat([x_att_buckets,pad_x],dim=2)
            y_att_buckets = torch.cat([y_att_buckets,pad_y],dim=2)
            fc_att_buckets = torch.cat([fc_att_buckets,pad_fc],dim=2)

        x_att_buckets = torch.reshape(x_att_buckets,(N,self.n_hashes,-1,self.chunk_size,C)) #[N, n_hashes, num_chunks, chunk_size, C] # q
        y_att_buckets = torch.reshape(y_att_buckets,(N,self.n_hashes,-1,self.chunk_size, C*self.reduction))
        fc_att_buckets = torch.reshape(fc_att_buckets,(N,self.n_hashes,-1,self.chunk_size, C*self.reduction))

        x_match = F.normalize(x_att_buckets, p=2, dim=-1,eps=5e-5)

        #allow attend to adjacent buckets
        x_match = self.add_adjacent_buckets(x_match) #[N, n_hashes, num_chunks, chunk_size*3, C]  # k
        y_att_buckets = self.add_adjacent_buckets(y_att_buckets)
        fc_att_buckets = self.add_adjacent_buckets(fc_att_buckets)
        fc_raw_score = self.fc(fc_att_buckets).permute(0,1,2,4,3) #[N, n_hashes, num_chunks, chunk_size, chunk_size*3]

        #unormalized attention score
        raw_score = torch.einsum('bhkie,bhkje->bhkij', x_att_buckets, x_match) + fc_raw_score #[N, n_hashes, num_chunks, chunk_size, chunk_size*3]

        #softmax    self.sigmoid2(self.fc2(self.sigmoid1(self.fc1(x_att_buckets))))
        bucket_score = torch.logsumexp(raw_score, dim=-1, keepdim=True)
        score = torch.exp(raw_score - bucket_score) #(after softmax)

        ret = torch.einsum('bukij,bukje->bukie', score, y_att_buckets) #[N, n_hashes, num_chunks, chunk_size, C*self.reduction]
        bucket_score = torch.reshape(bucket_score,[N,self.n_hashes,-1])
        ret = torch.reshape(ret,(N,self.n_hashes,-1,C*self.reduction))

        #if padded, then remove extra elements
        if padding:
            ret = ret[:,:,:-padding,:].clone()
            bucket_score = bucket_score[:,:,:-padding].clone()

        #recover the original order
        ret = torch.reshape(ret, (N, -1, C*self.reduction)) #[N, n_hashes*H*W,C]
        bucket_score = torch.reshape(bucket_score, (N, -1,)) #[N,n_hashes*H*W]
        ret = common.batched_index_select(ret, undo_sort)#[N, n_hashes*H*W,C]
        bucket_score = bucket_score.gather(1, undo_sort)#[N,n_hashes*H*W]

        #weighted sum multi-round attention
        ret = torch.reshape(ret, (N, self.n_hashes, L, C*self.reduction)) #[N, n_hashes*H*W,C]
        bucket_score = torch.reshape(bucket_score, (N, self.n_hashes, L, 1))
        probs = nn.functional.softmax(bucket_score,dim=1)
        ret = torch.sum(ret * probs, dim=1)

        ret = ret.permute(0,2,1).view(N,-1,H,W).contiguous()*self.res_scale+input
        return ret
