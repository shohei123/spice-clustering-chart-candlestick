import torch
import torch.nn as nn


def contrastive_loss(q, k, temperature):
    # normalize
    q = nn.functional.normalize(q, dim=1)
    k = nn.functional.normalize(k, dim=1)

    # gather all targets
    k = concat_all_gather(k)

    # Einstein sum is more intuitive
    # nはバッチサイズ、mはバッチサイズ×GPU合計数、cは分散表現の次元数
    logits = torch.einsum('nc,mc->nm', [q, k]) / temperature
    N = logits.shape[0]  # batch size per GPU

    # 1画像（分散表現）ごとに固有の整数ラベルを用意
    # バッチ数とGPUランクの乗算結果を加算により、ラベル重複しないように工夫
    labels = (torch.arange(N, dtype=torch.long) +
              N * torch.distributed.get_rank()).cuda()

    # 誤差逆伝播用に「nn」経由で損失を求めて、T乗算で調整後に返却
    return nn.CrossEntropyLoss()(logits, labels) * (2 * temperature)


@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [
        torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())
    ]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output
