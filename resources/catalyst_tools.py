import torch
from catalyst import runners, metrics, dl

from models import symmetric_npair_loss


class scRNARunner(runners.Runner):
    def handle_batch(self, batch):
        # model train/valid step
        # unpack the batch
        features_first = batch["features_first"].float()
        features_second = batch["features_second"].float()

        # run model forward pass
        logits, embeddings_first, embeddings_second = self.model(
            features_first, features_second
        )
        targets = torch.arange(logits.shape[0]).to(logits.device)

        # compute the loss
        loss = symmetric_npair_loss(logits, targets)

        # log metrics
        batch_temperature = self.model.logit_scale.exp().item()
        self.batch_metrics.update({"loss": loss})
        self.batch_metrics.update({"T": batch_temperature})

        self.batch = {
            "features_first": features_first,
            "features_second": features_second,
            "embeddings_first": embeddings_first,
            "embeddings_second": embeddings_second,
            "scores": logits,
            "targets": targets,
            "temperature": batch_temperature,
        }
        self.input = {
            "features_first": features_first,
            "features_second": features_second,
        }
        self.output = {
            "scores": logits,
            "embeddings_first": embeddings_first,
            "embeddings_second": embeddings_second,
        }

        # run model backward pass
        # print('self.is_train_loader', self.is_train_loader)
        if self.is_train_loader:
            self.engine.backward(loss)
            self.optimizer.step()
            self.optimizer.zero_grad()

    def get_loggers(self):
        return {
            "console": dl.ConsoleLogger(),
        }


class CustomMetric(metrics.ICallbackLoaderMetric):
    """Top1, Top5 accuracy metrics and competition score, without applying the matching algorithm."""

    def __init__(
            self, compute_on_call: bool = True, prefix: str = None, suffix: str = None
    ):
        super().__init__(compute_on_call=compute_on_call)
        self.prefix = prefix or ""
        self.suffix = suffix or ""
        self.embeddings_list_first = []
        self.embeddings_list_second = []

    def reset(self, num_batches: int, num_samples: int) -> None:
        self.embeddings_list_first = []
        self.embeddings_list_second = []
        torch.cuda.empty_cache()

    def update(self, *args, **kwargs) -> None:
        embeddings_first = kwargs["embeddings_first"]
        embeddings_second = kwargs["embeddings_second"]
        temperature = kwargs["temperature"]
        self.embeddings_list_first.append(temperature * embeddings_first)
        self.embeddings_list_second.append(embeddings_second)

    def compute(self):
        raise NotImplementedError("This method is not supported")

    def compute_key_value(self):
        all_embeddings_first = torch.cat(self.embeddings_list_first).detach().cpu()
        all_embeddings_second = torch.cat(self.embeddings_list_second).detach().cpu()
        logits = all_embeddings_first @ all_embeddings_second.T
        labels = torch.arange(logits.shape[0])

        del all_embeddings_first
        del all_embeddings_second

        forward_accuracy = (torch.argmax(logits, dim=1) == labels).float().mean().item()
        backward_accuracy = (
            (torch.argmax(logits, dim=0) == labels).float().mean().item()
        )

        logits_row_sums = logits.clip(min=0).sum(dim=1)
        top1_competition_metric = logits.clip(min=0).diag().div(logits_row_sums).mean().item()
        _, top_indexes_forward = logits.topk(5, dim=1)
        _, top_indexes_backward = logits.topk(5, dim=0)
        l_forward = labels.expand(5, logits.shape[0]).T
        l_backward = l_forward.T
        top5_forward_accuracy = (
            torch.any(top_indexes_forward == l_forward, 1).float().mean().item()
        )
        top5_backward_accuracy = (
            torch.any(top_indexes_backward == l_backward, 0).float().mean().item()
        )

        del logits

        avg_accuracy = 0.5 * (forward_accuracy + backward_accuracy)
        top5_avg_accuracy = 0.5 * (top5_forward_accuracy + top5_backward_accuracy)

        loader_metrics = {
            "forward_acc": forward_accuracy,
            "backward_acc": backward_accuracy,
            "avg_acc": avg_accuracy,
            "top5_forward_acc": top5_forward_accuracy,
            "top5_backward_acc": top5_backward_accuracy,
            "top5_avg_accuracy": top5_avg_accuracy,
            "top1_competition_metric": top1_competition_metric,
        }
        return loader_metrics
