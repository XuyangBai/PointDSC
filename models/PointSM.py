import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.common import knn, rigid_transform_3d
from utils.SE3 import transform


class NonLocalBlock(nn.Module):
    def __init__(self, num_channels=128, num_heads=1):
        super(NonLocalBlock, self).__init__()
        self.fc_message = nn.Sequential(
            nn.Conv1d(num_channels, num_channels//2, kernel_size=1),
            nn.BatchNorm1d(num_channels//2),
            nn.ReLU(inplace=True),
            nn.Conv1d(num_channels//2, num_channels//2, kernel_size=1),
            nn.BatchNorm1d(num_channels//2),
            nn.ReLU(inplace=True),
            nn.Conv1d(num_channels//2, num_channels, kernel_size=1),
        )
        self.projection_q = nn.Conv1d(num_channels, num_channels, kernel_size=1)
        self.projection_k = nn.Conv1d(num_channels, num_channels, kernel_size=1)
        self.projection_v = nn.Conv1d(num_channels, num_channels, kernel_size=1)
        self.num_channels = num_channels
        self.head = num_heads

    def forward(self, feat, attention):
        """
        Input:
            - feat:     [bs, num_channels, num_corr]  input feature
            - attention [bs, num_corr, num_corr]      spatial consistency matrix
        Output:
            - res:      [bs, num_channels, num_corr]  updated feature
        """
        bs, num_corr = feat.shape[0], feat.shape[-1]
        Q = self.projection_q(feat).view([bs, self.head, self.num_channels // self.head, num_corr])
        K = self.projection_k(feat).view([bs, self.head, self.num_channels // self.head, num_corr])
        V = self.projection_v(feat).view([bs, self.head, self.num_channels // self.head, num_corr])
        feat_attention = torch.einsum('bhco, bhci->bhoi', Q, K) / (self.num_channels // self.head) ** 0.5
        # combine the feature similarity with spatial consistency
        weight = torch.softmax(attention[:, None, :, :] * feat_attention, dim=-1)
        message = torch.einsum('bhoi, bhci-> bhco', weight, V).reshape([bs, -1, num_corr])
        message = self.fc_message(message)
        res = feat + message
        return res 


class NonLocalNet(nn.Module):
    def __init__(self, in_dim=6, num_layers=6, num_channels=128):
        super(NonLocalNet, self).__init__()
        self.num_layers = num_layers

        self.blocks = nn.ModuleDict()
        self.layer0 = nn.Conv1d(in_dim, num_channels, kernel_size=1, bias=True)
        for i in range(num_layers):
            layer = nn.Sequential(
                nn.Conv1d(num_channels, num_channels, kernel_size=1, bias=True),
                # nn.InstanceNorm1d(num_channels),
                nn.BatchNorm1d(num_channels),
                nn.ReLU(inplace=True)
            )
            self.blocks[f'PointCN_layer_{i}'] = layer
            self.blocks[f'NonLocal_layer_{i}'] = NonLocalBlock(num_channels)

    def forward(self, corr_feat, corr_compatibility):
        """
        Input: 
            - corr_feat:          [bs, in_dim, num_corr]   input feature map
            - corr_compatibility: [bs, num_corr, num_corr] spatial consistency matrix 
        Output:
            - feat:               [bs, num_channels, num_corr] updated feature
        """
        feat = self.layer0(corr_feat)
        for i in range(self.num_layers):
            feat = self.blocks[f'PointCN_layer_{i}'](feat)
            feat = self.blocks[f'NonLocal_layer_{i}'](feat, corr_compatibility)
        return feat


class PointSM(nn.Module):
    def __init__(self,
                 in_dim=6,
                 num_layers=6,
                 num_channels=128,
                 num_iterations=10,
                 ratio=0.1,
                 inlier_threshold=0.10,
                 sigma_d=0.10,
                 k=40,
                 nms_radius=0.10,
                 ):
        super(PointSM, self).__init__()
        self.num_iterations = num_iterations # maximum iteration of power iteration algorithm
        self.ratio = ratio # the maximum ratio of seeds.
        self.num_channels = num_channels
        self.inlier_threshold = inlier_threshold
        self.sigma = nn.Parameter(torch.Tensor([1.0]).float(), requires_grad=True)
        self.sigma_spat = nn.Parameter(torch.Tensor([sigma_d]).float(), requires_grad=False)
        self.k = k # neighborhood number in NSM module.
        self.nms_radius = nms_radius # only used during testing
        self.encoder = NonLocalNet(
            in_dim=in_dim,
            num_layers=num_layers,
            num_channels=num_channels,
        )

        self.classification = nn.Sequential(
            nn.Conv1d(num_channels, 32, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, 32, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, 1, kernel_size=1, bias=True),
        )

        # initialization
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.xavier_normal_(m.weight, gain=1)
            elif isinstance(m, (nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # add gradient clipping
        # grad_clip_norm = 100
        # for p in self.parameters():
        #     p.register_hook(lambda grad: torch.clamp(grad, -grad_clip_norm, grad_clip_norm))

    def forward(self, data):
        """
        Input:
            - corr_pos:   [bs, num_corr, 6]
            - src_keypts: [bs, num_corr, 3]
            - tgt_keypts: [bs, num_corr, 3]
            - testing:    flag for test phase, if False will not calculate M and post-refinement.
        Output: (dict)
            - final_trans:   [bs, 4, 4], the predicted transformation matrix. 
            - final_labels:  [bs, num_corr], the predicted inlier/outlier label (0,1), for classification loss calculation.
            - M:             [bs, num_corr, num_corr], feature similarity matrix, for SM loss calculation.
            - seed_trans:    [bs, num_seeds, 4, 4],  the predicted transformation matrix associated with each seeding point, deprecated.
            - corr_features: [bs, num_corr, num_channels], the feature for each correspondence, for circle loss calculation, deprecated.
            - confidence:    [bs], confidence of returned results, for safe guard, deprecated.
        """
        corr_pos, src_keypts, tgt_keypts = data['corr_pos'], data['src_keypts'], data['tgt_keypts']
        bs, num_corr = corr_pos.shape[0], corr_pos.shape[1]
        testing = 'testing' in data.keys()

        #################################
        # Step1: extract feature for each correspondence
        #################################
        with torch.no_grad():
            src_dist = torch.norm((src_keypts[:, :, None, :] - src_keypts[:, None, :, :]), dim=-1)
            corr_compatibility = src_dist - torch.norm((tgt_keypts[:, :, None, :] - tgt_keypts[:, None, :, :]), dim=-1)
            corr_compatibility = torch.clamp(1.0 - corr_compatibility ** 2 / self.sigma_spat ** 2, min=0)

        corr_features = self.encoder(corr_pos.permute(0,2,1), corr_compatibility).permute(0, 2, 1)
        normed_corr_features = F.normalize(corr_features, p=2, dim=-1)

        if not testing: # during training or validation
            # construct the feature similarity matrix M for loss calculation
            M = torch.matmul(normed_corr_features, normed_corr_features.permute(0, 2, 1))
            M = torch.clamp(1 - (1 - M) / self.sigma ** 2, min=0, max=1)
            # set diagnal of M to zero
            M[:, torch.arange(M.shape[1]), torch.arange(M.shape[1])] = 0
        else:
            M = None 

        #################################
        # Step 2.1: estimate initial confidence by MLP, find highly confident and well-distributed points as seeds.
        #################################
        # confidence = self.cal_leading_eigenvector(M.to(corr_pos.device), method='power')
        confidence = self.classification(corr_features.permute(0, 2, 1)).squeeze(1)
        
        if testing:
            seeds = self.pick_seeds(src_dist, confidence, R=self.nms_radius, max_num=int(num_corr * self.ratio))
        else:
            seeds = torch.argsort(confidence, dim=1, descending=True)[:, 0:int(num_corr * self.ratio)]
            
        
        #################################
        # Step 3 & 4: calculate transformation matrix for each seed, and find the best hypothesis.
        #################################
        seed_trans, seed_fitness, final_trans, final_labels = self.cal_seed_trans(seeds, normed_corr_features, src_keypts, tgt_keypts)

        # post refinement (only used during testing and bs == 1)
        if testing:
            final_trans = self.post_refinement(final_trans, src_keypts, tgt_keypts)

        ## during training, return the initial confidence as logits for classification loss
        ## during testing, return the final labels given by final transformation.
        if not testing:
            final_labels = confidence
        res = {
            "final_trans": final_trans, 
            "final_labels": final_labels, 
            "M": M
        }
        return res

    def pick_seeds(self, dists, scores, R, max_num):
        """
        Select seeding points using Non Maximum Suppression. (here we only support bs=1)
        Input:
            - dists:       [bs, num_corr, num_corr] src keypoints distance matrix
            - scores:      [bs, num_corr]     initial confidence of each correspondence
            - R:           float              radius of nms
            - max_num:     int                maximum number of returned seeds      
        Output:
            - picked_seeds: [bs, num_seeds]   the index to the seeding correspondences
        """
        assert scores.shape[0] == 1

        # parallel Non Maximum Suppression (more efficient)
        score_relation = scores.T >= scores  # [num_corr, num_corr], save the relation of leading_eig
        # score_relation[dists[0] >= R] = 1  # mask out the non-neighborhood node
        score_relation = score_relation.bool() | (dists[0] >= R).bool()
        is_local_max = score_relation.min(-1)[0].float()
        return torch.argsort(scores * is_local_max, dim=1, descending=True)[:, 0:max_num].detach()

        # # greedy Non Maximum Suppression
        # picked_seeds = []
        # selected_mask = torch.zeros_like(scores[0])
        # iter_num = 0
        # # if all the points are selected or the left points are all outlier, break
        # while torch.sum(selected_mask) != selected_mask.shape[0] and torch.sum(leading_eig[0] * (1 - selected_mask)) != 0:
        #     select_ind = torch.argmax(scores[0] * (1 - selected_mask))
        #     distance = torch.norm(src_keypts[0] - src_keypts[0, select_ind:select_ind + 1, :], dim=-1)
        #     selected_mask[distance < R] = 1
        #     picked_seeds.append(int(select_ind))
        #     iter_num += 1
        #     if iter_num > max_num:
        #         break
        # return torch.from_numpy(np.array(picked_seeds))[None, :].to(scores.device)

    def cal_seed_trans(self, seeds, corr_features, src_keypts, tgt_keypts):
        """
        Calculate the transformation for each seeding correspondences.
        Input: 
            - seeds:         [bs, num_seeds]              the index to the seeding correspondence
            - corr_features: [bs, num_corr, num_channels]
            - src_keypts:    [bs, num_corr, 3]
            - tgt_keypts:    [bs, num_corr, 3]
        Output: leading eigenvector
            - pairwise_trans:    [bs, num_seeds, 4, 4]  transformation matrix for each seeding point.
            - pairwise_fitness:  [bs, num_seeds]        fitness (inlier ratio) for each seeding point
            - final_trans:       [bs, 4, 4]             best transformation matrix (after post refinement) for each batch.
            - final_labels:      [bs, num_corr]         inlier/outlier label given by best transformation matrix.
        """
        bs, num_corr, num_channels = corr_features.shape[0], corr_features.shape[1], corr_features.shape[2]
        num_seeds = seeds.shape[-1]
        k = min(self.k, num_corr - 1)
        knn_idx = knn(corr_features, k=k, ignore_self=True, normalized=True)  # [bs, num_corr, k]
        knn_idx = knn_idx.gather(dim=1, index=seeds[:, :, None].expand(-1, -1, k))             # [bs, num_seeds, k]

        #################################
        # construct the feature consistency matrix of each correspondence subset.
        #################################
        knn_features = corr_features.gather(dim=1, index=knn_idx.view([bs, -1])[:, :, None].expand(-1, -1, num_channels)).view([bs, -1, k, num_channels])  # [bs, num_seeds, k, num_channels]
        knn_M = torch.matmul(knn_features, knn_features.permute(0, 1, 3, 2))
        knn_M = torch.clamp(1 - (1 - knn_M) / self.sigma ** 2, min=0)
        knn_M = knn_M.view([-1, k, k])
        feature_knn_M = knn_M

        #################################
        # construct the spatial consistency matrix of each correspondence subset.
        #################################
        src_knn = src_keypts.gather(dim=1, index=knn_idx.view([bs, -1])[:, :, None].expand(-1, -1, 3)).view([bs, -1, k, 3])  # [bs, num_seeds, k, 3]
        tgt_knn = tgt_keypts.gather(dim=1, index=knn_idx.view([bs, -1])[:, :, None].expand(-1, -1, 3)).view([bs, -1, k, 3])
        knn_M = ((src_knn[:, :, :, None, :] - src_knn[:, :, None, :, :]) ** 2).sum(-1) ** 0.5 - ((tgt_knn[:, :, :, None, :] - tgt_knn[:, :, None, :, :]) ** 2).sum(-1) ** 0.5
        # knn_M = torch.max(torch.zeros_like(knn_M), 1.0 - knn_M ** 2 / self.sigma_spat ** 2)
        knn_M = torch.clamp(1 - knn_M ** 2/ self.sigma_spat ** 2, min=0)
        knn_M = knn_M.view([-1, k, k])
        spatial_knn_M = knn_M

        #################################
        # Power iteratation to get the inlier probability
        #################################
        total_knn_M = feature_knn_M * spatial_knn_M
        total_knn_M[:, torch.arange(total_knn_M.shape[1]), torch.arange(total_knn_M.shape[1])] = 0
        # total_knn_M = self.gamma * feature_knn_M + (1 - self.gamma) * spatial_knn_M
        total_weight = self.cal_leading_eigenvector(total_knn_M, method='power')
        total_weight = total_weight.view([bs, -1, k])
        total_weight = total_weight / (torch.sum(total_weight, dim=-1, keepdim=True) + 1e-6)

        #################################
        # calculate the transformation by weighted least-squares for each subsets in parallel 
        #################################
        total_weight = total_weight.view([-1, k])

        src_knn = src_keypts.gather(dim=1, index=knn_idx.view([bs, -1])[:, :, None].expand(-1, -1, 3)).view([bs, -1, k, 3])  # [bs, num_seeds, k, 3]
        tgt_knn = tgt_keypts.gather(dim=1, index=knn_idx.view([bs, -1])[:, :, None].expand(-1, -1, 3)).view([bs, -1, k, 3])  # [bs, num_seeds, k, 3]
        src_knn, tgt_knn = src_knn.view([-1, k, 3]), tgt_knn.view([-1, k, 3])
        seed_as_center = False

        if seed_as_center:
            # if use seeds as the neighborhood centers
            src_center = src_keypts.gather(dim=1, index=seeds[:, :, None].expand(-1, -1, 3))  # [bs, num_seeds, 3]
            tgt_center = tgt_keypts.gather(dim=1, index=seeds[:, :, None].expand(-1, -1, 3))  # [bs, num_seeds, 3]
            src_center, tgt_center = src_center.view([-1, 3]), tgt_center.view([-1, 3])
            src_pts = src_knn[:, :, :, None] - src_center[:, None, :, None]  # [bs*num_seeds, k, 3, 1]
            tgt_pts = tgt_knn[:, :, :, None] - tgt_center[:, None, :, None]  # [bs*num_seeds, k, 3, 1]
            cov = torch.einsum('nkmo,nkop->nkmp', src_pts, tgt_pts.permute(0, 1, 3, 2))  # [bs*num_seeds, k, 3, 3]
            Covariances = torch.einsum('nkmp,nk->nmp', cov, total_weight)  # [bs*num_seeds, 3, 3]

            # use svd to recover the transformation for each seeding point, torch.svd is much faster on cpu.
            U, S, Vt = torch.svd(Covariances.cpu())
            U, S, Vt = U.cuda(), S.cuda(), Vt.cuda()
            delta_UV = torch.det(Vt @ U.permute(0, 2, 1))
            eye = torch.eye(3)[None, :, :].repeat(U.shape[0], 1, 1).to(U.device)
            eye[:, -1, -1] = delta_UV
            R = Vt @ eye @ U.permute(0, 2, 1)  # [num_pair, 3, 3]
            t = tgt_center[:, None, :] - src_center[:, None, :] @ R.permute(0, 2, 1)  # [num_pair, 1, 3]

            seedwise_trans = torch.eye(4)[None, :, :].repeat(R.shape[0], 1, 1).to(R.device)
            seedwise_trans[:, 0:3, 0:3] = R.permute(0, 2, 1)
            seedwise_trans[:, 0:3, 3:4] = t.permute(0, 2, 1)
            seedwise_trans = seedwise_trans.view([bs, -1, 4, 4])
        else:
            # not use seeds as neighborhood centers.
            seedwise_trans = rigid_transform_3d(src_knn, tgt_knn, total_weight)
            seedwise_trans = seedwise_trans.view([bs, -1, 4, 4])

        #################################
        # calculate the inlier number for each hypothesis, and find the best transformation for each point cloud pair
        #################################
        pred_position = torch.einsum('bsnm,bmk->bsnk', seedwise_trans[:, :, :3, :3], src_keypts.permute(0,2,1)) + seedwise_trans[:, :, :3, 3:4] # [bs, num_seeds, num_corr, 3]
        pred_position = pred_position.permute(0,1,3,2)
        L2_dis = torch.norm(pred_position - tgt_keypts[:, None, :, :], dim=-1)  # [bs, num_seeds, num_corr]
        seedwise_fitness = torch.mean((L2_dis < self.inlier_threshold).float(), dim=-1)  # [bs, num_seeds]
        # seedwise_inlier_rmse = torch.sum(L2_dis * (L2_dis < config.inlier_threshold).float(), dim=1)
        batch_best_guess = seedwise_fitness.argmax(dim=1)

        # refine the pose by using all the inlier correspondences (done in the post-refinement step)
        final_trans = seedwise_trans.gather(dim=1, index=batch_best_guess[:, None, None, None].expand(-1, -1, 4, 4)).squeeze(1)
        final_labels = L2_dis.gather(dim=1, index=batch_best_guess[:, None, None].expand(-1, -1, L2_dis.shape[2])).squeeze(1)
        final_labels = (final_labels < self.inlier_threshold).float()  
        return seedwise_trans, seedwise_fitness, final_trans, final_labels

    def cal_leading_eigenvector(self, M, method='power'):
        """
        Calculate the leading eigenvector using power iteration algorithm or torch.symeig
        Input: 
            - M:      [bs, num_corr, num_corr] the compatibility matrix 
            - method: select different method for calculating the learding eigenvector.
        Output: 
            - solution: [bs, num_corr] leading eigenvector
        """
        if method == 'power':
            # power iteration algorithm
            leading_eig = torch.ones_like(M[:, :, 0:1])
            leading_eig_last = leading_eig
            for i in range(self.num_iterations):
                leading_eig = torch.bmm(M, leading_eig)
                leading_eig = leading_eig / (torch.norm(leading_eig, dim=1, keepdim=True) + 1e-6)
                if torch.allclose(leading_eig, leading_eig_last):
                    break
                leading_eig_last = leading_eig
            leading_eig = leading_eig.squeeze(-1)
            return leading_eig
        elif method == 'eig':  # cause NaN during back-prop
            e, v = torch.symeig(M, eigenvectors=True)
            leading_eig = v[:, :, -1]
            return leading_eig
        else:
            exit(-1)

    def cal_confidence(self, M, leading_eig, method='eig_value'):
        """
        Calculate the confidence of the spectral matching solution based on spectral analysis.
        Input: 
            - M:          [bs, num_corr, num_corr] the compatibility matrix 
            - leading_eig [bs, num_corr]           the leading eigenvector of matrix M
        Output: 
            - confidence  
        """
        if method == 'eig_value':
            # max eigenvalue as the confidence (Rayleigh quotient)
            max_eig_value = (leading_eig[:, None, :] @ M @ leading_eig[:, :, None]) / (leading_eig[:, None, :] @ leading_eig[:, :, None])
            confidence = max_eig_value.squeeze(-1)
            return confidence
        elif method == 'eig_value_ratio':
            # max eigenvalue / second max eigenvalue as the confidence
            max_eig_value = (leading_eig[:, None, :] @ M @ leading_eig[:, :, None]) / (leading_eig[:, None, :] @ leading_eig[:, :, None])
            # compute the second largest eigen-value
            B = M - max_eig_value * leading_eig[:, :, None] @ leading_eig[:, None, :]
            solution = torch.ones_like(B[:, :, 0:1])
            for i in range(self.num_iterations):
                solution = torch.bmm(B, solution)
                solution = solution / (torch.norm(solution, dim=1, keepdim=True) + 1e-6)
            solution = solution.squeeze(-1)
            second_eig = solution
            second_eig_value = (second_eig[:, None, :] @ B @ second_eig[:, :, None]) / (second_eig[:, None, :] @ second_eig[:, :, None])
            confidence = max_eig_value / second_eig_value
            return confidence
        elif method == 'xMx':
            # max xMx as the confidence (x is the binary solution)
            # rank = torch.argsort(leading_eig, dim=1, descending=True)[:, 0:int(M.shape[1]*self.ratio)]
            # binary_sol = torch.zeros_like(leading_eig)
            # binary_sol[0, rank[0]] = 1
            confidence = leading_eig[:, None, :] @ M @ leading_eig[:, :, None]
            confidence = confidence.squeeze(-1) / M.shape[1]
            return confidence

    def post_refinement(self, initial_trans, src_keypts, tgt_keypts, weights=None):
        """
        Perform post refinement using the initial transformation matrix, only adopted during testing.
        Input
            - initial_trans: [bs, 4, 4] 
            - src_keypts:    [bs, num_corr, 3]    
            - tgt_keypts:    [bs, num_corr, 3]
            - weights:       [bs, num_corr]
        Output:    
            - final_trans:   [bs, 4, 4]
        """
        assert initial_trans.shape[0] == 1
        if self.inlier_threshold == 0.10: # for 3DMatch
            inlier_threshold_list = [0.10] * 20
        else: # for KITTI
            inlier_threshold_list = [1.2] * 20

        previous_inlier_num = 0
        for inlier_threshold in inlier_threshold_list:
            warped_src_keypts = transform(src_keypts, initial_trans)
            L2_dis = torch.norm(warped_src_keypts - tgt_keypts, dim=-1)
            pred_inlier = (L2_dis < inlier_threshold)[0]  # assume bs = 1
            inlier_num = torch.sum(pred_inlier)
            if abs(int(inlier_num - previous_inlier_num)) < 1:
                break
            else:
                previous_inlier_num = inlier_num
            initial_trans = rigid_transform_3d(
                A=src_keypts[:, pred_inlier, :],
                B=tgt_keypts[:, pred_inlier, :],
                ## https://link.springer.com/article/10.1007/s10589-014-9643-2
                # weights=None,
                weights=1/(1 + (L2_dis/inlier_threshold)**2)[:, pred_inlier],
                # weights=((1-L2_dis/inlier_threshold)**2)[:, pred_inlier]
            )
        return initial_trans
