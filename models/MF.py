import torch
import torch.nn as nn

class MF(nn.Module):
    """
        Manifold fitting implementation with pytorch
    """
    def __init__(self, X, r0, r1, r2, k=3.0):
        """Initialize the MF class.
        Args:
            X: torch.Tensor, shape (n, p, ...), sample data tensor
            r0: float, the radius for calculating the contraction direction, in the order of O(sigma * log(1/sigma)))
            r1: float, the first radius for constructing the contraction region, in the order of O(sigma)
            r2: float, the second radius for calculating the contraction region, in the order of O(sigma * log(1/sigma)))
            k: float, the power parameter for calculating the weight
        """
        super(MF, self).__init__()
        self.out_shape = X[0].shape
        if len(self.out_shape) > 1:
            self.Is_pic = True
            X = X.reshape(X.shape[0], -1)
        else:
            self.Is_pic = False

        self.X = X
        self.r0_init = r0
        self.r1_init = r1
        self.r2_init = r2
        self.r0 = nn.Parameter(torch.tensor(r0), requires_grad=True)
        self.r1 = nn.Parameter(torch.tensor(r1), requires_grad=True)
        self.r2 = nn.Parameter(torch.tensor(r2), requires_grad=True)
        self.k = k
    
    def weight1(self,dist,r):
        w = torch.zeros_like(dist)
        # check whether the radius is NA
        if r.isnan():
            w = torch.ones_like(dist)
        else:
            flag1 = dist < r
            w[flag1] = (1 - (dist[flag1]/r)**2)**self.k

        return w
    
    def weight2(self,dist,r):
        w = torch.zeros_like(dist)  # Initialize tensor w with zeros

        flag1 = dist < r / 2
        flag2 = (dist >= r / 2) & (dist < r)

        w[flag1] = 1.0
        w[flag2] = (1 - ((2 * dist[flag2] - r) / r) ** 2) ** self.k

        return w
    
    def Normalize_weight(self, w):
        return w / torch.sum(w, dim=0, keepdim=True)

    def forward(self, Z):
        """Compute the manifold fitting function G(Z).
        Args:
            Z: torch.Tensor, shape (n, p, ...), new data tensor
        """
        if self.Is_pic == True:
            Z = Z.reshape(Z.shape[0], -1)

        if self.r0 > self.r0_init*10:
            self.r0 = nn.Parameter(torch.tensor(self.r0_init*10), requires_grad=True)
        if self.r1 > self.r1_init*10:
            self.r1 = nn.Parameter(torch.tensor(self.r1_init*10), requires_grad=True)
        if self.r2 > self.r2_init*10:
            self.r2 = nn.Parameter(torch.tensor(self.r2_init*10), requires_grad=True)


        dist = torch.cdist(self.X, Z) # Compute the distance between X and Z, shape (N, n)

        inds = torch.any(dist <= self.r0, dim=1).nonzero().squeeze() # Find the indices of X that are within the contraction region of Z
        dist = dist[inds] # Select the distance between X and Z within the contraction region, shape (N, n)
        X = self.X[inds] # Select the data within the contraction region, shape (N, p)
        alpha = self.weight1(dist, self.r0)
        alpha = self.Normalize_weight(alpha)

        # Compute mu: the F(z) function for each z
        mu = torch.matmul(alpha.t(), X)
        flag0 = torch.isnan(mu)
        mu[flag0] = Z[flag0] # If all weights are zero, set mu as Z

        U = Z - mu 
        U = torch.unsqueeze(U, dim=1) # Add a dimension to U to have shape (n, 1, p)
        # UTU = torch.div(torch.matmul(U.transpose(1, 2),U), torch.matmul(U,U.transpose(1, 2))) # Compute UU^T

        diff_vectors = torch.unsqueeze(X, dim=1) - torch.unsqueeze(Z, dim=0) # Compute X - Z, shape (N, n, p)
        diff_vectors = torch.unsqueeze(diff_vectors, dim=2) # Add a dimension to have shape (N, n, 1, p)
        # projection = torch.matmul(diff_vectors, UTU) # Compute the projection of X - Z onto U, shape (N, n, 1, p)
        projection = torch.matmul(diff_vectors, U.transpose(1, 2)) # Compute the projection of X - Z onto U, shape (N, n, 1, 1)
        projection = torch.matmul(projection, U) # Compute the projection of X - Z onto U, shape (N, n, 1, p)

        dist_u = torch.matmul(projection,projection.transpose(2, 3)) # Compute the squared distance between X and Z along U, shape (N, n, 1, 1)
        dist_u = torch.squeeze(dist_u) # Squeeze dist_u to have shape (N, n)
        dist_v = (dist**2 - dist_u)**0.5 # Compute the distance between X and Z along V, shape (N, n)
        dist_u = dist_u**0.5 # Compute the distance between X and Z along U, shape (N, n)

        beta = self.weight2(dist_v, self.r1) * self.weight2(dist_u, self.r2)
        beta = self.Normalize_weight(beta)

        # Compute the manifold fitting function G(Z) for each z
        e_Z = torch.matmul(beta.t(), X)
        flag0 = torch.isnan(e_Z)
        e_Z[flag0] = Z[flag0] # If all weights are zero, set e as Z

        if self.Is_pic:
            e_Z = e_Z.reshape(e_Z.shape[0], *self.out_shape)


        return e_Z

        # if not self.Is_pic and self.X.shape[1] <200:
        #     dist = torch.cdist(self.X, Z) # Compute the distance between X and Z, shape (N, n)

        #     inds = torch.any(dist <= self.r0, dim=1).nonzero().squeeze() # Find the indices of X that are within the contraction region of Z
        #     dist = dist[inds] # Select the distance between X and Z within the contraction region, shape (N, n)
        #     X = self.X[inds] # Select the data within the contraction region, shape (N, p)
        #     alpha = self.weight1(dist, self.r0)
        #     alpha = self.Normalize_weight(alpha)

        #     # Compute mu: the F(z) function for each z
        #     mu = torch.matmul(alpha.t(), X)
        #     flag0 = torch.isnan(mu)
        #     mu[flag0] = Z[flag0] # If all weights are zero, set mu as Z

        #     U = Z - mu 
        #     U = torch.unsqueeze(U, dim=1) # Add a dimension to U to have shape (n, 1, p)
        #     UTU = torch.div(torch.matmul(U.transpose(1, 2),U), torch.matmul(U,U.transpose(1, 2))) # Compute UU^T

        #     diff_vectors = torch.unsqueeze(X, dim=1) - torch.unsqueeze(Z, dim=0) # Compute X - Z, shape (N, n, p)
        #     diff_vectors = torch.unsqueeze(diff_vectors, dim=2) # Add a dimension to have shape (N, n, 1, p)
        #     projection = torch.matmul(diff_vectors, UTU) # Compute the projection of X - Z onto U, shape (N, n, 1, p)

        #     dist_u = torch.matmul(projection,projection.transpose(2, 3)) # Compute the squared distance between X and Z along U, shape (N, n, 1, 1)
        #     dist_u = torch.squeeze(dist_u) # Squeeze dist_u to have shape (N, n)
        #     dist_v = (dist**2 - dist_u)**0.5 # Compute the distance between X and Z along V, shape (N, n)
        #     dist_u = dist_u**0.5 # Compute the distance between X and Z along U, shape (N, n)

        #     beta = self.weight2(dist_v, self.r1) * self.weight2(dist_u, self.r2)
        #     beta = self.Normalize_weight(beta)

        #     # Compute the manifold fitting function G(Z) for each z
        #     e_Z = torch.matmul(beta.t(), X)
        #     flag0 = torch.isnan(e_Z)
        #     e_Z[flag0] = Z[flag0] # If all weights are zero, set e as Z
        # else:
        #     Z = Z.reshape(Z.shape[0], -1)
        #     e_Z = Z * 0.0 # Initialize e_Z with zeros
        #     for i in range(Z.shape[0]):
        #         z = Z[i]
        #         dist = torch.cdist(self.X, z.unsqueeze(0)) # Compute the distance between X and Z, shape (N, n)
        #         inds = torch.any(dist <= self.r0, dim=1).nonzero().squeeze() # Find the indices of X that are within the contraction region of Z
        #         if inds == None:
        #             e_Z[i] = z
        #             continue
        #         dist = dist[inds] # Select the distance between X and Z within the contraction region, shape (N, n)
        #         X = self.X[inds] # Select the data within the contraction region, shape (N, p)
        #         alpha = self.weight1(dist, self.r0)
        #         alpha = self.Normalize_weight(alpha)
        #         alpha = alpha.reshape(-1, 1) # Reshape alpha to have shape (N, 1)
        #         X = X.reshape(-1, self.X.shape[1]) # Reshape X to have shape (N, p)

        #         # Compute mu: the F(z) function for each z
        #         mu = torch.matmul(alpha.t(), X).reshape(-1, self.X.shape[1])

        #         U = z - mu
        #         if torch.sum(U**2) == 0:
        #             e_Z[i] = z
        #             continue
        #         U = torch.unsqueeze(U, dim=1) # Add a dimension to U to have shape (1, 1, p)
        #         UTU = torch.div(torch.matmul(U.transpose(1, 2),U), torch.matmul(U,U.transpose(1, 2))) # Compute UU^T

        #         diff_vectors = torch.unsqueeze(X, dim=1) - torch.unsqueeze(z, dim=0) # Compute X - Z, shape (N, 1, p)
        #         diff_vectors = torch.unsqueeze(diff_vectors, dim=2) # Add a dimension to have shape (N, 1, 1, p)
        #         projection = torch.matmul(diff_vectors, UTU) # Compute the projection of X - Z onto U, shape (N, 1, 1, p)

        #         dist_u = torch.matmul(projection,projection.transpose(2, 3)) # Compute the squared distance between X and Z along U, shape (N, 1, 1, 1)
        #         dist_u = torch.squeeze(dist_u) # Squeeze dist_u to have shape (N, 1)
        #         dist_v = (dist**2 - dist_u)**0.5 # Compute the distance between X and Z along V, shape (N, 1)
        #         dist_u = dist_u**0.5 # Compute the distance between X and Z along U, shape (N, 1)

        #         beta = self.weight2(dist_v, self.r1) * self.weight2(dist_u, self.r2)
        #         beta = self.Normalize_weight(beta)
        #         beta = beta.reshape(-1, 1) # Reshape beta to have shape (N, 1)

        #         # Compute the manifold fitting function G(Z) for each z
        #         e_z = torch.matmul(beta.t(), X)
        #         e_z = e_z.reshape(-1, self.X.shape[1])
        #         if e_z.isnan().any():
        #             e_z = z
        #         e_Z[i] = e_z
        #         del dist, inds, X, alpha, mu, U, UTU, diff_vectors, projection, dist_u, dist_v, beta, e_z
        #     e_Z = e_Z.reshape(-1, *self.out_shape)

