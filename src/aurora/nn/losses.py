from aurora.autograd import AutoNode


def mean_squared_error(pred: list[AutoNode], trgt: list[AutoNode]):
    tot_loss = AutoNode(value=0)
    assert len(pred) == len(trgt)
    for p, t in zip(pred, trgt):
        loss = (p - t) ** 2
        tot_loss += loss
    return tot_loss / AutoNode(value=len(pred))
