import torch
import torch.nn as nn

def chunk_batched_delta_rule_forward(Q,K,V,beta,C):
    """
        Q,K and V are of shape (B,L,d) where B is the batch, L the sequence length and d the embeding's dimension
        beta is of shape (B,L,1)
        C is the size of the chunk, it should divide L
    """
    B,L,d = Q.shape
    Q, K, V= map(lambda x: x.reshape(B,-1,C,d), [Q,K,V])
    beta = beta.reshape(B,-1,C)
    K_beta = K*beta.unsqueeze(-1)
    V_beta = V*beta.unsqueeze(-1)

    mask = torch.triu(torch.ones(C,C), diagonal=0).bool()

    K_t = torch.transpose(K,2,3)
    T = -(K_beta[:] @ K_t[:]).masked_fill(mask,0)
   
   #forward substitution 
    for k in range(L//C):
        for i in range(1,C):
            T_new = T.clone()
            T_new[:,k,i,:i] = T[:,k,i,:i] + (T[:,k,i,:,None]*T[:,k,:,:i]).sum(-2)
            T = T_new
        T[:,k] = T[:,k] + torch.eye(C)

    W = T @ K_beta
    U = T @ V_beta

    S = torch.zeros((B,d,d))
    O = torch.empty_like(V)
    mask = torch.triu(torch.ones(C,C), diagonal=1).bool()

    for i in range(L//C):
        q_i, k_i, w_i = Q[:,i], K[:,i], W[:,i]
        u_i = U[:,i]-w_i@S
        o_inter = q_i @ S
        A_i = (q_i @ k_i.transpose(1,2)).masked_fill(mask,0)
        o_intra = A_i @ u_i
        S = S + k_i.transpose(1,2)@u_i
        O[:,i] = o_intra + o_inter
        
    return O.reshape(B,L,d)

def delta_rule_recurrent_step(q_t, k_t, v_t, beta_t, S_prev):
    """
    Perform a single step of the recurrent Delta Rule.
    
    Args:
        q_t: Query vector at time step t, shape (d,).
        k_t: Key vector at time step t, shape (d,).
        v_t: Value vector at time step t, shape (d,).
        beta_t: Writing strength scalar at time step t, shape ().
        S_prev: Previous hidden state (memory matrix), shape (d, d).
        
    Returns:
        o_t: Output vector at time step t, shape (d,).
        S_new: Updated hidden state (memory matrix), shape (d, d).
    """
    # Compute old value
    v_old_t = S_prev @ k_t  # Shape (d,)
    
    # Compute new value
    v_new_t = beta_t * v_t + (1 - beta_t) * v_old_t  # Shape (d,)
    
    # Update hidden state (memory)
    S_new = S_prev - torch.outer(v_old_t, k_t) + torch.outer(v_new_t, k_t)  # Shape (d, d)
    
    # Compute output
    o_t = S_new @ q_t  # Shape (d,)
    
    return o_t, S_new

class DeltaBlock(nn.Module):
    def __init__(self,d,expand=1, neg_eigen=False):
        """
            d is the dimension of the input
            d*expand is the size of the hidden state
            neg_eigen if true allow the model to have negative eigen value. It was not on the original paper but on another: https://arxiv.org/abs/2411.12537.
        """
        super(DeltaBlock,self).__init__()
        self.d = d
        self.expand = expand
        self.Wq = nn.Linear(d,d*expand)
        self.Wk = nn.Linear(d,d*expand)
        self.Wv = nn.Linear(d,d*expand)

        self.proj_out = nn.Linear(d*expand,d)

        self.beta = nn.Linear(d,1)
        self.sigma = nn.Sigmoid()
        self.alpha = 2 if neg_eigen else 1

    def forward(self,X,chunk=1):
        """
            this is the chunkwise form of deltanet
            input: 
                X of shape B,L,d
                chunk size
            output: Y of shape B,L,d
        """
        if chunk ==1:
            _,chunk,_ = X.shape        
        return self.proj_out(chunk_batched_delta_rule_forward(self.Wq(X),self.Wk(X),self.Wv(X)/self.alpha,self.alpha*self.sigma(self.beta(X)),chunk))

    def step(self,X,S=None):
        """
            this is the parallel form of deltanet
            input:
                X vector of shape d
                S state of shape (d,d), if not provided, the model will initialize it with zeros(0,0)
            output:
                Y vector of shape d
                S new state of shape (d,d)
        """
        if S==None:
            S = torch.zeros(self.d*self.expand,self.d*self.expand)
        y,S = delta_rule_recurrent_step(self.Wq(X),self.Wk(X),self.Wv(X)/self.alpha,self.alpha*self.sigma(self.beta(X)),S)
        return self.proj_out(y), S
