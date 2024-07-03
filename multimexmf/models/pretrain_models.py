from M3L.models.pretrain_models import VTMAE as M3LVTMAE
import torch
import torch.nn as nn
from typing import Tuple, Dict, Optional, Type, Any
from stable_baselines3.common.type_aliases import Schedule

EPS = 1e-6


class VTMAE(M3LVTMAE):
    def __init__(self, patch_agg: str = 'max_min', *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.patch_agg = patch_agg

    def aggregate_patches(self, x: dict) -> dict:
        if self.patch_agg == 'mean':
            return {k: torch.mean(v, dim=-2) for k, v in x.items()}
        elif self.patch_agg == 'max_min':
            x = {k: torch.cat([torch.max(v, dim=-2).values[..., None], torch.min(v, dim=-2).values[..., None]], dim=-1)
                 for k, v in x.items()}
            return {k: v.reshape((*v.shape[:-2], -1)) for k, v in x.items()}
        elif self.patch_agg == 'all':
            return {k: v.reshape((*v.shape[:-2], -1)) for k, v in x.items()}
        else:
            raise NotImplementedError

    def get_observation_emb(self, x: dict, eval: bool = True, use_vision: bool = True, use_tactile: bool = True) \
            -> dict:

        if eval:
            self.eval()
        else:
            self.train()
        if 'image' in x.keys():
            batch = x['image'].shape[0]
            device = x['image'].device
        elif 'tactile' in x.keys():
            batch = x['tactile1'].shape[0]
            device = x['tactile1'].device
            use_vision = False
        else:
            raise AssertionError

        if self.early_conv_masking:
            if use_vision:
                image_tokens = self.early_conv_vision(x['image'])
            else:
                image_tokens = torch.zeros((batch, 0, self.encoder_dim)).to(device)
            if self.num_tactiles > 0 and use_tactile:
                tactile_tokens_list = []
                for i in range(self.num_tactiles):
                    tactile_tokens_list.append(self.early_conv_tactile(x['tactile' + str(i + 1)]))
                tactile_tokens = torch.cat(tactile_tokens_list, dim=1)
            else:
                tactile_tokens = torch.zeros((batch, 0, image_tokens.shape[-1])).to(device)

        else:
            if use_vision:
                image_patches = self.image_to_patch(x['image'])
                batch, num_image_patches, *_ = image_patches.shape
                image_tokens = self.image_patch_to_emb(image_patches)
            else:
                image_tokens = torch.zeros((batch, 0, self.encoder_dim)).to(device)

            if self.num_tactiles > 0 and use_tactile:
                tactile_patches_list = []
                for i in range(self.num_tactiles):
                    tactile_patches_list.append(self.tactile_to_patch(x['tactile' + str(i + 1)]))
                tactile_patches = torch.cat(tactile_patches_list, dim=1)
                batch, num_tactile_patches, *_ = tactile_patches.shape
                tactile_tokens = self.tactile_patch_to_emb(tactile_patches)
            else:
                tactile_tokens = torch.zeros((batch, 0, self.encoder_dim)).to(device)

        embeddings = {
            'image': image_tokens,
            'tactile': tactile_tokens,
        }
        return self.aggregate_patches(embeddings)


class MultiHeadGaussianEnsemble(nn.Module):
    def __init__(self, input_dim: int, output_dict: Dict, num_heads: int = 5,
                 features: Tuple = (256, 256, 256),
                 act_fn: nn.Module = nn.ReLU(), learn_std: bool = False, min_std: float = 1e-3, max_std: float = 1e2,
                 disagreement_weights: Optional[Dict] = None, use_entropy: bool = True, agg_disag: str = 'sum',
                 optimizer_class: Type[torch.optim.Optimizer] = torch.optim.Adam,
                 optimizer_kwargs: Optional[Dict[str, Any]] = None,
                 ):
        super().__init__()
        self.output_dict = output_dict
        self.learn_std = learn_std
        self.min_std = min_std
        self.max_std = max_std
        self.num_heads = num_heads
        self.input_dim = input_dim
        self.output_shapes = {k: v.shape[-1] for k, v in output_dict.items()}
        modules = [
            nn.Sequential(
                nn.Linear(input_dim, features[0]),
                act_fn,
            )
        ]
        prev_size = features[0]
        for i in range(1, len(features)):
            modules.append(
                nn.Sequential(
                    nn.Linear(prev_size, features[i]),
                    act_fn,
                )
            )
            prev_size = features[i]

        self.feat = nn.ModuleList(modules)
        if self.learn_std:
            output_modules = [[key, nn.Linear(prev_size, 2 * num_heads * shape)]
                              for key, shape in self.output_shapes.items()]
        else:
            output_modules = [[key, nn.Linear(prev_size, num_heads * shape)]
                              for key, shape in self.output_shapes.items()]
        self.output_modules = nn.ModuleDict(output_modules)
        if disagreement_weights is not None:
            assert disagreement_weights.keys() == output_dict.keys()
            self.disagreement_weights = disagreement_weights
        else:
            self.disagreement_weights = {k: 1.0 for k in output_dict.keys()}

        self.use_entropy = use_entropy
        self.agg_disag = agg_disag

        if optimizer_kwargs is None:
            optimizer_kwargs = {}
            # Small values to avoid NaN in Adam optimizer
            if optimizer_class == torch.optim.Adam:
                optimizer_kwargs["eps"] = 1e-5

        self.optimizer = optimizer_class(self.parameters(), **optimizer_kwargs)

    def forward(self, x) -> Dict:
        batch_size = x.shape[0]
        for feat in self.feat:
            x = feat(x)

        output_dict = {}
        for key, shape in self.output_shapes.items():
            pred = self.output_modules[key](x).reshape(batch_size, -1, self.num_heads)
            if self.learn_std:
                mean, log_std = torch.split(pred, shape, dim=-2)
                std = nn.functional.softplus(log_std)
                std = torch.clamp(std, min=self.min_std, max=self.max_std)
                output_dict[key] = (mean, std)
            else:
                output_dict[key] = pred
        return output_dict

    def loss(self, prediction: Dict, target: Dict):
        loss = {}
        for key, val in target.items():
            if self.learn_std:
                assert isinstance(prediction[key], Tuple)
                # dim: batch, feature, num_ensemble
                mean, std = prediction[key]
                loss[key] = - (torch.distributions.Normal(mean, std).log_prob(val[..., None]).mean())
            else:
                mean = prediction[key]
                loss[key] = - (torch.distributions.Normal(mean, torch.ones_like(mean)).log_prob(val[..., None]).mean())
        return loss

    def get_disagreement(self, prediction: Dict) -> torch.Tensor:
        disagreement = {}
        for key, val in prediction.items():
            if self.learn_std:
                assert isinstance(prediction[key], Tuple)
                mean, std = val
                assert mean.shape[-1] == self.num_heads and std.shape[-1] == self.num_heads
                epistemic_std = mean.std(dim=-1)
                al_std = std.mean(dim=-1)
                ratio = torch.square(epistemic_std / al_std)
                if self.use_entropy:
                    # take mean over output dim
                    disagreement[key] = torch.log(1 + ratio).mean(dim=-1) * self.disagreement_weights[key]
                else:
                    # take mean over batch dim
                    disagreement[key] = ratio.mean(-1) * self.disagreement_weights[key]
            else:
                assert val.shape[-1] == self.num_heads
                epistemic_var = torch.square(val.std(dim=-1))
                if self.use_entropy:
                    # take mean over output dim
                    disagreement[key] = torch.log(EPS + epistemic_var).mean(dim=-1) * self.disagreement_weights[key]
                else:
                    # take mean over batch dim
                    disagreement[key] = epistemic_var.mean(dim=-1) * self.disagreement_weights[key]

        total_disagreements = torch.stack([val.reshape(-1, 1) for val in disagreement.values()], dim=-1)
        if self.agg_disag == 'max':
            return total_disagreements.max(dim=-1)
        elif self.agg_disag == 'sum':
            return total_disagreements.sum(dim=-1)
        else:
            raise NotImplementedError


class SimpleMLP(nn.Module):
    def __init__(self,
                 input_dim: int,
                 output_dim: Optional[int] = None,
                 features: Tuple = (256, 256, 256),
                 act_fn: nn.Module = nn.ReLU(),
                 learn_std: bool = False,
                 ):
        super().__init__()
        self.learn_std = learn_std
        modules = [
            nn.Sequential(
                nn.Linear(input_dim, features[0]),
                act_fn,
            )
        ]
        prev_size = features[0]
        for i in range(1, len(features)):
            modules.append(
                nn.Sequential(
                    nn.Linear(prev_size, features[i]),
                    act_fn,
                )
            )
            prev_size = features[i]
        self.output_dim = output_dim
        if output_dim:
            if self.learn_std:
                modules.append(nn.Linear(prev_size, 2 * output_dim))
            else:
                modules.append(nn.Linear(prev_size, output_dim))

        self.feat = nn.ModuleList(modules)

    def forward(self, x) -> Dict:
        for feat in self.feat:
            x = feat(x)
        return x


class EnsembleMLP(nn.Module):
    def __init__(self, input_dim: int, output_dict: Dict, num_heads: int = 5,
                 features: Tuple = (256, 256, 256),
                 act_fn: nn.Module = nn.ReLU(), learn_std: bool = False, min_std: float = 1e-3, max_std: float = 1e2,
                 disagreement_weights: Optional[Dict] = None, use_entropy: bool = True, agg_disag: str = 'sum',
                 optimizer_class: Type[torch.optim.Optimizer] = torch.optim.Adam,
                 optimizer_kwargs: Optional[Dict[str, Any]] = None,
                 ):
        super().__init__()
        self.learn_std = learn_std
        self.num_heads = num_heads
        self.output_shapes = {k: v.shape[-1] for k, v in output_dict.items()}
        self.min_std = min_std
        self.max_std = max_std
        output_dim = sum(self.output_shapes.values())
        self.models = nn.ModuleList([SimpleMLP(
            input_dim=input_dim,
            features=features,
            output_dim=output_dim,
            act_fn=act_fn,
            learn_std=learn_std,
        ) for _ in range(num_heads)])

        if disagreement_weights is not None:
            assert disagreement_weights.keys() == output_dict.keys()
            self.disagreement_weights = disagreement_weights
        else:
            self.disagreement_weights = {k: 1.0 for k in output_dict.keys()}

        self.use_entropy = use_entropy
        self.agg_disag = agg_disag
        self.output_dict = output_dict

        if optimizer_kwargs is None:
            optimizer_kwargs = {}
            # Small values to avoid NaN in Adam optimizer
            if optimizer_class == torch.optim.Adam:
                optimizer_kwargs["eps"] = 1e-5

        self.optimizer = optimizer_class(self.parameters(), **optimizer_kwargs)

    def forward(self, x) -> Dict:
        # concatenate outputs of the ensemble
        outputs = torch.cat([model(x)[..., None] for model in self.models], dim=-1)
        if self.learn_std:
            # split by 2 x output size if learning stds
            outputs = torch.split(outputs, [2 * val for val in self.output_shapes.values()], dim=-2)
        else:
            outputs = torch.split(outputs, [val for val in self.output_shapes.values()], dim=-2)
        output_dict = {}
        for i, (key, val) in enumerate(self.output_shapes.items()):
            if self.learn_std:
                mean, log_std = torch.split(outputs[i], val, dim=-2)
                std = nn.functional.softplus(log_std)
                std = torch.clamp(std, min=self.min_std, max=self.max_std)
                output_dict[key] = (mean, std)
            else:
                output_dict[key] = outputs[i]
        return output_dict

    def loss(self, prediction: Dict, target: Dict):
        loss = {}
        for key, val in target.items():
            if self.learn_std:
                assert isinstance(prediction[key], Tuple)
                # dim: batch, feature, num_ensemble
                mean, std = prediction[key]
                loss[key] = - (torch.distributions.Normal(mean, std).log_prob(val[..., None]).mean())
            else:
                mean = prediction[key]
                loss[key] = - (torch.distributions.Normal(mean, torch.ones_like(mean)).log_prob(val[..., None]).mean())
        return loss

    def get_disagreement(self, prediction: Dict) -> torch.Tensor:
        disagreement = {}
        for key, val in prediction.items():
            if self.learn_std:
                assert isinstance(prediction[key], Tuple)
                mean, std = val
                assert mean.shape[-1] == self.num_heads and std.shape[-1] == self.num_heads
                epistemic_std = mean.std(dim=-1)
                al_std = std.mean(dim=-1)
                ratio = torch.square(epistemic_std / al_std)
                if self.use_entropy:
                    # take mean over output dim
                    disagreement[key] = torch.log(1 + ratio).mean(dim=-1) * self.disagreement_weights[key]
                else:
                    # take mean over batch dim
                    disagreement[key] = ratio.mean(-1) * self.disagreement_weights[key]
            else:
                assert val.shape[-1] == self.num_heads
                epistemic_var = torch.square(val.std(dim=-1))
                if self.use_entropy:
                    # take mean over output dim
                    disagreement[key] = torch.log(EPS + epistemic_var).mean(dim=-1) * self.disagreement_weights[key]
                else:
                    # take mean over batch dim
                    disagreement[key] = epistemic_var.mean(dim=-1) * self.disagreement_weights[key]

        total_disagreements = torch.stack([val.reshape(-1, 1) for val in disagreement.values()], dim=-1)
        if self.agg_disag == 'max':
            return total_disagreements.max(dim=-1)
        elif self.agg_disag == 'sum':
            return total_disagreements.sum(dim=-1)
        else:
            raise NotImplementedError


if __name__ == '__main__':
    learn_std = False
    from torch.utils.data import TensorDataset, DataLoader
    import matplotlib.pyplot as plt

    input_dim = 1
    output_dim = 2

    noise_level = 0.01
    d_l, d_u = 0, 10
    xs = torch.linspace(d_l, d_u, 32).reshape(-1, 1)
    ys = torch.concatenate([torch.sin(xs), torch.cos(xs)], dim=1)
    ys = ys + noise_level * torch.randn(size=ys.shape)
    train_loader = DataLoader(TensorDataset(xs, ys), shuffle=True, batch_size=32)
    model = EnsembleMLP(input_dim=1, output_dict={'y': ys}, features=(256, 256),
                                      optimizer_kwargs={'lr': 1e-3, 'weight_decay': 1e-4}, num_heads=5,
                                      learn_std=learn_std)

    n_epochs = 1000
    n_steps = 0
    for i in range(n_epochs):
        for X_batch, Y_batch in train_loader:
            model.train()
            model.optimizer.zero_grad()
            predictions = model(X_batch)
            loss = model.loss(predictions, target={'y': Y_batch})
            total_loss = torch.stack([val.reshape(-1, 1) for val in loss.values()], dim=-1).mean()
            total_loss.backward()
            model.optimizer.step()

    model.eval()
    test_xs = torch.linspace(-5, 15, 1000).reshape(-1, 1)
    test_ys = torch.concatenate([torch.sin(test_xs), torch.cos(test_xs)], dim=1)

    if learn_std:
        y_pred, _ = model(test_xs)['y']
    else:
        y_pred = model(test_xs)['y']

    xs = xs.cpu().numpy()
    ys = ys.cpu().numpy()

    test_xs = test_xs.cpu().numpy()
    test_ys = test_ys.cpu().numpy()
    y_pred = y_pred.cpu().detach().numpy()
    scale = 2
    for j in range(output_dim):
        plt.scatter(xs.reshape(-1), ys[:, j], label='Data', color='red')
        for i in range(model.num_heads):
            plt.plot(test_xs, y_pred[:, j, i], label='NN prediction', color='black', alpha=0.3)
        y_mean = y_pred.mean(axis=-1)
        eps_std = y_pred.std(axis=-1)
        plt.plot(test_xs, y_mean[..., j], label='Mean', color='blue')
        plt.fill_between(test_xs.reshape(-1),
                         (y_mean[..., j] - scale * eps_std[..., j]).reshape(-1),
                         (y_mean[..., j] + scale * eps_std[..., j]).reshape(-1),
                         label=r'$2\sigma$', alpha=0.3, color='blue')
        handles, labels = plt.gca().get_legend_handles_labels()
        plt.plot(test_xs.reshape(-1), test_ys[:, j], label='True', color='green')
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys())
        plt.show()
