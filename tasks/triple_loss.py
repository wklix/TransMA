import torch
from torch.nn import functional as F
from torch import nn

eps = 1e-8  # an arbitrary small value to be used for numerical stability tricks


def euclidean_distance_matrix(x):
    """Efficient computation of Euclidean distance matrix
    Args:
      x: Input tensor of shape (batch_size, embedding_dim)

    Returns:
      Distance matrix of shape (batch_size, batch_size)
    """


    # step 1 - compute the dot product
    # shape: (batch_size, batch_size)
    dot_product = torch.mm(x, x.t())
    # step 2 - extract the squared Euclidean norm from the diagonal
    # shape: (batch_size,)
    squared_norm = torch.diag(dot_product)
    # step 3 - compute squared Euclidean distances
    # shape: (batch_size, batch_size)
    distance_matrix = squared_norm.unsqueeze(0) - 2 * dot_product + squared_norm.unsqueeze(1)
    # get rid of negative distances due to numerical instabilities
    distance_matrix = F.relu(distance_matrix)
    # step 4 - compute the non-squared distances
    # handle numerical stability
    # derivative of the square root operation applied to 0 is infinite
    # we need to handle by setting any 0 to eps
    mask = (distance_matrix == 0.0).float()
    # use this mask to set indices with a value of 0 to eps
    mask_1= mask * eps
    distance_matrix = distance_matrix+mask_1
    # now it is safe to get the square root
    distance_matrix = torch.sqrt(distance_matrix)
    # undo the trick for numerical stability
    #distance_matrix *= (1.0 - mask)
    distance_matrix = distance_matrix *(1.0 - mask)
    return distance_matrix



def get_triplet_mask(labels):
  """compute a mask for valid triplets
  Args:
    labels: Batch of integer labels. shape: (batch_size,)
  Returns:
    Mask tensor to indicate which triplets are actually valid. Shape: (batch_size, batch_size, batch_size)
    A triplet is valid if:
    `labels[i] == labels[j] and labels[i] != labels[k]`
    and `i`, `j`, `k` are different.
  """
  # step 1 - get a mask for distinct indices
  # shape: (batch_size, batch_size)
  indices_equal = torch.eye(labels.size()[0], dtype=torch.bool, device=labels.device)
  indices_not_equal = torch.logical_not(indices_equal)
  # shape: (batch_size, batch_size, 1)
  i_not_equal_j = indices_not_equal.unsqueeze(2)
  # shape: (batch_size, 1, batch_size)
  i_not_equal_k = indices_not_equal.unsqueeze(1)
  # shape: (1, batch_size, batch_size)
  j_not_equal_k = indices_not_equal.unsqueeze(0)
  # Shape: (batch_size, batch_size, batch_size)
  distinct_indices = torch.logical_and(torch.logical_and(i_not_equal_j, i_not_equal_k), j_not_equal_k)
  # step 2 - get a mask for valid anchor-positive-negative triplets
  # shape: (batch_size, batch_size)
  labels_equal = labels.unsqueeze(0) == labels.unsqueeze(1)
  # shape: (batch_size, batch_size, 1)
  i_equal_j = labels_equal.unsqueeze(2)
  # shape: (batch_size, 1, batch_size)
  i_equal_k = labels_equal.unsqueeze(1)
  # shape: (batch_size, batch_size, batch_size)
  valid_indices = torch.logical_and(i_equal_j, torch.logical_not(i_equal_k))
  # step 3 - combine two masks
  mask = torch.logical_and(distinct_indices, valid_indices)
  return mask


class BatchAllTtripletLoss(nn.Module):
    """Uses all valid triplets to compute Triplet loss
    Args:
      margin: Margin value in the Triplet Loss equation
    """

    def __init__(self, margin=0.3):
        super().__init__()
        self.margin = margin

    def forward(self, embeddings, labels):
        """computes loss value.
        Args:
          embeddings: Batch of embeddings, e.g., output of the encoder. shape: (batch_size, embedding_dim)
          labels: Batch of integer labels associated with embeddings. shape: (batch_size,)
        Returns:
          Scalar loss value.
        """
        # step 1 - get distance matrix
        # shape: (batch_size, batch_size)
        distance_matrix = euclidean_distance_matrix(embeddings)
        # step 2 - compute loss values for all triplets by applying broadcasting to distance matrix
        # shape: (batch_size, batch_size, 1)
        anchor_positive_dists = distance_matrix.unsqueeze(2)
        # shape: (batch_size, 1, batch_size)
        anchor_negative_dists = distance_matrix.unsqueeze(1)
        # get loss values for all possible n^3 triplets
        # shape: (batch_size, batch_size, batch_size)
        triplet_loss = anchor_positive_dists - anchor_negative_dists + self.margin
        # step 3 - filter out invalid or easy triplets by setting their loss values to 0
        # shape: (batch_size, batch_size, batch_size)
        mask = get_triplet_mask(labels)
        #triplet_loss *= mask
        triplet_loss =  triplet_loss *mask
        # easy triplets have negative loss values
        triplet_loss = F.relu(triplet_loss)

        # step 4 - compute scalar loss value by averaging positive losses
        num_positive_losses = (triplet_loss > eps).float().sum()
        triplet_loss = triplet_loss.sum() / (num_positive_losses + eps)

        return triplet_loss


class BatchAllTtripletLoss_multi_module_version(nn.Module):
    """Uses all valid triplets to compute Triplet loss
    Args:
      margin: Margin value in the Triplet Loss equation
    """

    def __init__(self, margin=0.3):
        super().__init__()
        self.margin = margin

    def forward(self, output1, output2, weight=None):
        """computes loss value.
            output1: text output batch x embedding_dim
            output2: image output batch x embedding_dim
            weight: 1-tanimoto similarity between smiles batch x batch

            if weight == None: plain version
            if weight != None: weight punished version
        """
        output1=nn.functional.normalize(output1, dim=1)
        output2=nn.functional.normalize(output2, dim=1)
        cur_batch_size = output1.size(0)
        # step 0 - construct a big vector
        embeddings = torch.cat((output1, output2), dim=0) # 2 * batch x embedding_dim
        # construct labels for the big vector 0
        labels = torch.cat((torch.arange(cur_batch_size), torch.arange(cur_batch_size)), dim=0).to(output1.device) # 2 * batch

        # step 1 - get distance matrix
        # shape: (2*batch_size, 2*batch_size)
        distance_matrix = euclidean_distance_matrix(embeddings)
        # step 2 - compute loss values for all triplets by applying broadcasting to distance matrix
        # shape: (2*batch_size, 2*batch_size, 1)
        anchor_positive_dists = distance_matrix.unsqueeze(2)
        # shape: (2*batch_size, 1, 2*batch_size)
        anchor_negative_dists = distance_matrix.unsqueeze(1)
        # get loss values for all possible n^3 triplets
        # shape: (2*batch_size, 2*batch_size, 2*batch_size)
        triplet_loss = anchor_positive_dists - anchor_negative_dists + self.margin
        # step 3 - filter out invalid or easy triplets by setting their loss values to 0
        # shape: (2*batch_size, 2*batch_size, 2*batch_size)
        mask = get_triplet_mask(labels)
        #triplet_loss *= mask
        triplet_loss = triplet_loss *mask
        # easy triplets have negative loss values
        triplet_loss = F.relu(triplet_loss)

        # step 4 - compute scalar loss value by averaging positive losses
        num_positive_losses = (triplet_loss > eps).float().sum()
        if weight is not None:
            # if weight == None: plain version
            # if weight != None: weight punished version
            punished_matrix = torch.zeros((2*cur_batch_size, 2*cur_batch_size), dtype=weight.dtype, device=weight.device)
            # Fill the four quadrants
            punished_matrix[:cur_batch_size, :cur_batch_size] = weight  # Top-left
            punished_matrix[cur_batch_size:, :cur_batch_size] = weight  # Bottom-left
            punished_matrix[:cur_batch_size, cur_batch_size:] = weight  # Top-right
            punished_matrix[cur_batch_size:, cur_batch_size:] = weight  # Bottom-right
            punished_matrix = punished_matrix.unsqueeze(1)   # 2*batch x 1 x 2*batch
            #triplet_loss *= punished_matrix
            triplet_loss = triplet_loss *punished_matrix
            num_positive_losses = (triplet_loss > eps).float().sum()
        triplet_loss = triplet_loss.sum() / (num_positive_losses + eps)

        return triplet_loss
