import torch


def triplet_loss(anchor, positive, negative, margin=1e-5):
    """
    Naive implementation of the triplet loss function

    Returns:
    loss -- value of the loss
    """

    # Distance between the anchor and the positive
    pos_dist = torch.mean(torch.pow(anchor - positive,2))

    # Distance between the anchor and the negative
    neg_dist = torch.mean(torch.pow(anchor - negative,2))

    # Compute triplet loss
    basic_loss = pos_dist - neg_dist + margin
    trip_loss = torch.maximum(basic_loss, torch.zeros_like(basic_loss))

    return trip_loss

def norm(tensor):
    tensor = torch.mean(tensor, dim=1, keepdims=True)
    tensor = (tensor - torch.mean(tensor)) / torch.std(tensor)

    return tensor

def trip_loss(out1, out2, out3, out4,hr, diver_w = 1e-4, margin_same=1e-5, margin_diff=2e-5, decay_rate = 0.8):

    # Pos zero
    zero_map = torch.zeros_like(hr)

    # Norm
    out1_mean = norm(out1)
    out2_mean = norm(out2)
    out3_mean = norm(out3)
    out4_mean = norm(out4)
    hr_mean = norm(hr)

    # Compute divergence loss
    trip_loss12 = triplet_loss(torch.abs(out1_mean - hr_mean), zero_map, torch.abs(out2_mean - hr_mean), margin=margin_same) * decay_rate
    trip_loss13 = triplet_loss(torch.abs(out1_mean - hr_mean), zero_map, torch.abs(out3_mean - hr_mean), margin=margin_diff)
    trip_loss14 = triplet_loss(torch.abs(out1_mean - hr_mean), zero_map, torch.abs(out4_mean - hr_mean), margin=margin_diff)

    trip_loss21 = triplet_loss(torch.abs(out2_mean - hr_mean), zero_map, torch.abs(out1_mean - hr_mean), margin=margin_same) * decay_rate
    trip_loss23 = triplet_loss(torch.abs(out2_mean - hr_mean), zero_map, torch.abs(out3_mean - hr_mean), margin=margin_diff)
    trip_loss24 = triplet_loss(torch.abs(out2_mean - hr_mean), zero_map, torch.abs(out4_mean - hr_mean), margin=margin_diff)

    trip_loss31 = triplet_loss(torch.abs(out3_mean - hr_mean), zero_map, torch.abs(out1_mean - hr_mean), margin=margin_diff)
    trip_loss32 = triplet_loss(torch.abs(out3_mean - hr_mean), zero_map, torch.abs(out2_mean - hr_mean), margin=margin_diff)
    trip_loss34 = triplet_loss(torch.abs(out3_mean - hr_mean), zero_map, torch.abs(out4_mean - hr_mean), margin=margin_same) * decay_rate


    trip_loss41 = triplet_loss(torch.abs(out4_mean - hr_mean), zero_map, torch.abs(out1_mean - hr_mean), margin=margin_diff)
    trip_loss42 = triplet_loss(torch.abs(out4_mean - hr_mean), zero_map, torch.abs(out2_mean - hr_mean), margin=margin_diff)
    trip_loss43 = triplet_loss(torch.abs(out4_mean - hr_mean), zero_map, torch.abs(out3_mean - hr_mean), margin=margin_same) * decay_rate

    loss_trip = diver_w * (
                trip_loss12 + trip_loss13 + trip_loss14 + trip_loss21 + trip_loss23 + trip_loss24 +
                trip_loss31 + trip_loss32 + trip_loss34 + trip_loss41 + trip_loss42 + trip_loss43) / 12

    return loss_trip

if __name__ == '__main__':
    pass