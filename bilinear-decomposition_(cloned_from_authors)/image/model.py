import torch
from torch import nn, Tensor
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset

from transformers import PretrainedConfig, PreTrainedModel
from jaxtyping import Float
from tqdm import tqdm
from pandas import DataFrame
from einops import *

from shared.components import Linear, Bilinear

from image.adversarial_attack import *

def _collator(transform=None):
    def inner(batch):
        x = torch.stack([item[0] for item in batch]).float()
        y = torch.stack([item[1] for item in batch])
        return (x, y) if transform is None else (transform(x), y)
    return inner

class Config(PretrainedConfig):
    def __init__(
        self,
        lr: float = 1e-3,
        wd: float = 0.5,
        epochs: int = 100,
        batch_size: int = 2048,
        d_hidden: int = 256,
        n_layer: int = 1,
        d_input: int = 784,
        d_output: int = 10,
        bias: bool = False,
        residual: bool = False,
        seed: int = 42,
        **kwargs
    ):
        self.lr = lr
        self.wd = wd
        self.epochs = epochs
        self.batch_size = batch_size
        self.seed = seed
    
        self.d_hidden = d_hidden
        self.n_layer = n_layer
        self.d_input = d_input
        self.d_output = d_output
        self.bias = bias
        self.residual = residual
        
        
        
        super().__init__(**kwargs)


class Model(PreTrainedModel):
    def __init__(self, config) -> None:
        super().__init__(config)
        torch.manual_seed(config.seed)
        
        d_input, d_hidden, d_output = config.d_input, config.d_hidden, config.d_output
        bias, n_layer = config.bias, config.n_layer
        
        self.embed = Linear(d_input, d_hidden, bias=False)
        self.blocks = nn.ModuleList([Bilinear(d_hidden, d_hidden, bias=bias) for _ in range(n_layer)])
        self.head = Linear(d_hidden, d_output, bias=False)
        
        self.criterion = nn.CrossEntropyLoss()
        self.accuracy = lambda y_hat, y: (y_hat.argmax(dim=-1) == y).float().mean()
    
    def forward(self, x: Float[Tensor, "... inputs"]) -> Float[Tensor, "... outputs"]:
        x = self.embed(x.flatten(start_dim=1))
        
        for layer in self.blocks:
            x = x + layer(x) if self.config.residual else layer(x)
        
        return self.head(x)
    
    @property
    def w_e(self):
        return self.embed.weight.data
    
    @property
    def w_u(self):
        return self.head.weight.data
    
    @property
    def w_lr(self):
        return torch.stack([rearrange(layer.weight.data, "(s o) h -> s o h", s=2) for layer in self.blocks])
    
    @property
    def w_l(self):
        return self.w_lr.unbind(1)[0]
    
    @property
    def w_r(self):
        return self.w_lr.unbind(1)[1]
    
    @classmethod
    def from_config(cls, *args, **kwargs):
        return cls(Config(*args, **kwargs))

    @classmethod
    def from_pretrained(cls, path, *args, **kwargs):
        new = cls(Config(*args, **kwargs))
        new.load_state_dict(torch.load(path))
        return new
    
    def step(self, x, y):
        y_hat = self(x)
        
        loss = self.criterion(y_hat, y)
        accuracy = self.accuracy(y_hat, y)
        
        return loss, accuracy

    def fit(self, train, test, transform=None, defense_strategy=None):
        torch.manual_seed(self.config.seed)
        torch.set_grad_enabled(True)

        optimizer = AdamW(self.parameters(), lr=self.config.lr, weight_decay=self.config.wd)
        scheduler = CosineAnnealingLR(optimizer, T_max=self.config.epochs)

        loader = DataLoader(train, batch_size=self.config.batch_size, shuffle=True, drop_last=True,
                            collate_fn=_collator(transform))
        test_x, test_y = test.x, test.y

        pbar = tqdm(range(self.config.epochs))
        history = []

        for _ in pbar:
            epoch = []
            for x, y in loader:
                if defense_strategy == None:
                    loss, acc = self.train().step(x, y)
                    epoch += [(loss.item(), acc.item())]

                elif defense_strategy == "FGSM":
                    loss, logits = fgsm_loss(
                        model=self,
                        criterion=self.criterion,
                        inputs=x,
                        labels=y
                    )
                    acc = self.accuracy(logits, y)
                    epoch += [(loss.item(), acc.item())]

                elif defense_strategy == "PGD":
                    # Get adverserial examples using PGD attack
                    adversarial_inputs = pgd_attack(
                        model=self,
                        data=x,
                        target=y,
                        criterion=self.criterion
                    )

                    # Add them to the original batch
                    inputs = torch.cat([x, adversarial_inputs], dim=0)
                    # Make sure the model has the correct labels
                    labels = torch.cat([y, y], dim=0)

                    optimizer.zero_grad()
                    outputs = self(inputs)
                    loss = self.criterion(outputs, labels)
                    acc = self.accuracy(outputs, labels)

                    epoch += [(loss.item(), acc.item())]

                else:
                    raise ValueError(f"An unknown defense strategy encountered: {defense_strategy}")

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            scheduler.step()

            val_loss, val_acc = self.eval().step(test_x, test_y)

            metrics = {
                "train/loss": sum(loss for loss, _ in epoch) / len(epoch),
                "train/acc": sum(acc for _, acc in epoch) / len(epoch),
                "val/loss": val_loss.item(),
                "val/acc": val_acc.item()
            }

            history.append(metrics)
            pbar.set_description(', '.join(f"{k}: {v:.3f}" for k, v in metrics.items()))

        torch.set_grad_enabled(False)

        return DataFrame.from_records(history, columns=['train/loss', 'train/acc', 'val/loss', 'val/acc'])

    def fit_cifar(self, train, test, transform=None):
        torch.manual_seed(self.config.seed)
        torch.set_grad_enabled(True)

        device = next(self.parameters()).device

        optimizer = AdamW(self.parameters(), lr=self.config.lr, weight_decay=self.config.wd)
        scheduler = CosineAnnealingLR(optimizer, T_max=self.config.epochs)

        if hasattr(train, 'x') and hasattr(test, 'y'):
            train_loader = DataLoader(TensorDataset(train.x, train.y),
                                      batch_size=self.config.batch_size, shuffle=True)
        else:
            train_loader = DataLoader(train, batch_size=self.config.batch_size, shuffle=True)
        if hasattr(test, 'x') and hasattr(test, 'y'):
            test_loader = DataLoader(TensorDataset(test.x, test.y),
                                     batch_size=self.config.batch_size, shuffle=False)
        else:
            test_loader = DataLoader(test, batch_size=self.config.batch_size, shuffle=False)

        pbar = tqdm(range(self.config.epochs))
        history = []

        for _ in pbar:
            self.train()
            epoch = []
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                loss, acc = self.step(x, y)
                epoch += [(loss.item(), acc.item())]

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            scheduler.step()

            self.eval()
            val_epoch = []

            with torch.no_grad():
                for x, y in test_loader:
                    x, y = x.to(device), y.to(device)
                    loss, acc = self.step(x, y)
                    val_epoch += [(loss.item(), acc.item())]

            metrics = {
                "train/loss": sum(loss for loss, _ in epoch) / len(epoch),
                "train/acc": sum(acc for _, acc in epoch) / len(epoch),
                "val/loss": sum(loss for loss, _ in val_epoch) / len(val_epoch),
                "val/acc": sum(acc for _, acc in val_epoch) / len(val_epoch),
            }

            history.append(metrics)
            pbar.set_description(', '.join(f"{k}: {v:.3f}" for k, v in metrics.items()))

        torch.set_grad_enabled(False)
        return DataFrame.from_records(history, columns=['train/loss', 'train/acc', 'val/loss', 'val/acc'])

    def decompose(self):
        """The function to decompose a single-layer model into eigenvalues and eigenvectors."""
        
        # Split the bilinear layer into the left and right components
        l, r = self.w_lr[0].unbind()
        
        # Compute the third-order (bilinear) tensor
        b = einsum(self.w_u, l, r, "cls out, out in1, out in2 -> cls in1 in2")
        
        # Symmetrize the tensor
        b = 0.5 * (b + b.mT)

        # Perform the eigendecomposition
        vals, vecs = torch.linalg.eigh(b)
        
        # Project the eigenvectors back to the input space
        vecs = einsum(vecs, self.w_e, "cls emb comp, emb inp -> cls comp inp")
        
        # Return the eigenvalues and eigenvectors
        return vals, vecs