from __future__ import annotations

import pytest
import torch

from module_a.src.models.aam_softmax import AAMSoftmax, AAMSoftmaxError


def test_aam_logits_and_loss_are_finite():
    head = AAMSoftmax(192, 3, margin=0.2, scale=30.0)
    embeddings = torch.randn(4, 192)
    labels = torch.tensor([0, 1, 2, 1])

    logits = head(embeddings, labels)
    loss = head.loss(embeddings, labels)

    assert logits.shape == (4, 3)
    assert loss.ndim == 0
    assert torch.isfinite(logits).all()
    assert torch.isfinite(loss)


def test_aam_backward_reaches_embeddings_and_head():
    head = AAMSoftmax(8, 2)
    embeddings = torch.randn(3, 8, requires_grad=True)
    loss = head.loss(embeddings, torch.tensor([0, 1, 0]))
    loss.backward()

    assert embeddings.grad is not None
    assert head.weight.grad is not None
    assert torch.isfinite(head.weight.grad).all()


@pytest.mark.parametrize(
    "labels",
    [torch.tensor([0.0, 1.0]), torch.tensor([0, 3]), torch.tensor([[0], [1]])],
)
def test_aam_rejects_invalid_label_contract(labels):
    with pytest.raises(AAMSoftmaxError):
        AAMSoftmax(8, 3)(torch.randn(2, 8), labels)


def test_aam_rejects_non_finite_embeddings():
    embeddings = torch.randn(2, 8)
    embeddings[0, 0] = float("nan")
    with pytest.raises(AAMSoftmaxError, match="NaN"):
        AAMSoftmax(8, 2)(embeddings, torch.tensor([0, 1]))

