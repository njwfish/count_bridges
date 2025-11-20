import torch
from torch.utils.data import Dataset
from torch.distributions import Dirichlet
from typing import Tuple, Dict, Any, Optional
import numpy as np
from .gaussian_mixture import GaussianMixtureDataset, LowRankGaussianMixtureDataset
from .mnist import DiffMNIST

class DeconvolutionGaussianMixtureDataset(Dataset):
    """
    Deconvolution dataset that wraps LowRankGaussianMixtureDataset.
    
    This dataset:
    1. Uses LowRankGaussianMixtureDataset as the underlying data source
    2. Provides one-hot context vectors indicating mixture component
    3. Uses Dirichlet sampling to determine component mixing in groups
    4. Samples actual group members from the appropriate mixture components
    
    Each item returns:
      x_0: Group of source samples from Gaussian mixture [group_size, data_dim]
      x_1: Group of target samples from Gaussian mixture [group_size, data_dim] 
      z: One-hot matrix indicating component for each group member [group_size, k]
      X_0: Sum of group members [data_dim]
      
    The key innovation is that x_0/X_0 contains actual samples from the dataset, with the number
    of samples from each mixture component determined by Dirichlet-sampled weights.
    """
    def __init__(
        self,
        size: int,
        base_size: int,
        data_dim: int,
        min_value: int,
        value_range: int,
        context_dim: int,  # This should equal k (number of mixture components)
        group_size: int = 10,
        latent_dim: int = None,  # For low-rank variant
        k: int = 5,  # Number of mixture components
        dirichlet_concentration: Optional[float] = None,  # Concentration parameter for Dirichlet
        seed: int = 42,
        uniform_lambda: float = 0.0,
        fraction_unit_level: float = 0.0,
        **kwargs  # Additional args passed to LowRankGaussianMixtureDataset
    ):
        super().__init__()
        self.size = size
        self.base_size = base_size
        self.data_dim = data_dim
        self.min_value = min_value
        self.value_range = value_range
        self.group_size = group_size
        self.k = k
        self.dirichlet_concentration = dirichlet_concentration
        self.uniform_lambda = uniform_lambda
        self.fraction_unit_level = fraction_unit_level
        
        # Use low-rank variant if latent_dim is specified, otherwise use regular variant
        if latent_dim is not None:
            self.mixture_dataset = LowRankGaussianMixtureDataset(
                size=base_size,
                data_dim=data_dim,
                latent_dim=latent_dim,
                value_range=value_range,
                min_value=min_value,
                k=k,
                seed=seed,
                **kwargs
            )
        else:
            self.mixture_dataset = GaussianMixtureDataset(
                size=base_size,
                data_dim=data_dim,
                value_range=value_range,
                min_value=min_value,
                k=k,
                seed=seed,
                **kwargs
            )
        
        # Pre-compute all groups directly from mixture components
        self._pre_compute_all_groups(seed)
        self.use_unit_level_idx = np.random.randint(0, self.size, np.round(self.fraction_unit_level * self.size).astype(int))

            
    def _pre_compute_all_groups(self, seed: int):
        """Pre-compute all groups by directly sampling from mixture components"""
        # Use torch's random state for consistent seeding
        with torch.random.fork_rng():
            torch.manual_seed(seed + 1000)  # Different seed to avoid conflicts
            
            # Sample Dirichlet weights for component allocation
            # Shape: [size, k] - each row sums to 1, represents component proportions
            base_component_weights = self.uniform_lambda * self.mixture_dataset.weights_source + (1 - self.uniform_lambda) * torch.ones(self.k) / self.k
            if self.dirichlet_concentration is not None:
                dirichlet_dist = Dirichlet(self.dirichlet_concentration * base_component_weights)
                component_weights = dirichlet_dist.sample((self.size,))
            else:
                component_weights = base_component_weights.expand(self.size, -1)
                        
            self.component_indices = {}
            for component_id in range(self.k):
                component_mask = (self.mixture_dataset.x0_components == component_id)
                self.component_indices[component_id] = torch.where(component_mask)[0]
            
            # Pre-allocate storage for all groups and their component info
            self.groups = torch.zeros(self.size, self.group_size, self.data_dim, dtype=torch.float32)
            self.group_component_info = torch.zeros(self.size, self.group_size, dtype=torch.int64)
            self.group_one_hot = torch.zeros(self.size, self.group_size, self.k, dtype=torch.float32)
            
            all_component_assignments = torch.multinomial(
                component_weights.view(-1, self.k),  # [size, k]
                self.group_size, 
                replacement=True
            )  # [size, group_size]
            
            # Store component assignments
            self.group_component_info = all_component_assignments
            
            self.group_one_hot = torch.zeros(self.size, self.group_size, self.k, dtype=torch.float32)
            self.group_one_hot.scatter_(2, all_component_assignments.unsqueeze(-1).long(), 1.0)
            
            # Flatten all assignments to process in batches by component
            flat_assignments = all_component_assignments.flatten()  # [size * group_size]
            flat_samples = torch.zeros(self.size * self.group_size, self.data_dim, dtype=torch.float32)
            
            # Process each component in batch
            for component_id in range(self.k):
                # Find all positions that need this component
                component_mask = (flat_assignments == component_id)
                n_needed = component_mask.sum().item()
                
                if n_needed > 0:
                    # Get pre-computed indices for this component
                    component_indices = self.component_indices[component_id]
                    
                    # Randomly select n_needed samples (with replacement)
                    selected_indices = component_indices[torch.randint(0, len(component_indices), (n_needed,))]
                    selected_samples = self.mixture_dataset.x0_data[selected_indices].float()
                    
                    # Assign to the appropriate positions
                    flat_samples[component_mask] = selected_samples
            
            # Reshape back to groups
            self.groups = flat_samples.view(self.size, self.group_size, self.data_dim)
            
            print(f"Pre-computed all {self.size} groups with vectorized operations")

    def __len__(self): 
        return self.size

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]: 
        # Get random target sample (same as before)
        target_idx = np.random.randint(0, self.size, self.group_size)
        x_1 = self.mixture_dataset.x1_data[target_idx]  # [group_size, data_dim]
        
        # Get pre-computed group as both x_0 and X_0
        x_0 = self.groups[idx]  # [group_size, data_dim]

        # Get pre-computed one-hot vectors for each group member
        z = self.group_one_hot[idx]  # [group_size, k] - one-hot for each member


        if idx in self.use_unit_level_idx:
            # select random unit level idx
            unit_level_idx = np.random.randint(0, self.group_size, 1)
            x_1 = x_1[unit_level_idx]
            x_0 = x_0[unit_level_idx]  # [group_size, data_dim]
            z = z[unit_level_idx]
            
        X_0 = x_0.sum(dim=0)  # [data_dim] - sum of group members
        
        return {
            'x_0': x_0.float(),  # [group_size, data_dim]
            'x_1': x_1.float(),  # [group_size, data_dim] 
            'z': z.float(),      # [group_size, k] - one-hot for each member
            'X_0': X_0.float()   # [data_dim] - sum of group members
        }

class DeconvolutionMNISTGaussianMixtureDataset(DeconvolutionGaussianMixtureDataset):
    """
    Deconvolution dataset that combines MNIST images with Gaussian mixture data.
    
    This dataset extends DeconvolutionGaussianMixtureDataset by:
    1. Using MNIST images selected based on mixture component assignments
    2. Providing structured x_0/x_1 with both 'img' (MNIST) and 'counts' (Gaussian) components
    3. Maintaining the same group-based structure for deconvolution
    
    Each item returns:
      x_0: Dict with 'img' and 'counts' components [group_size, ...]
      x_1: Dict with 'img' (noise) and 'counts' (Gaussian) [group_size, ...]
      z: One-hot matrix indicating component for each group member [group_size, k]
      X_0: Sum of Gaussian counts [data_dim]
    """
    
    def __init__(
        self,
        **kwargs
    ):
        """
        Args:
            **kwargs: Arguments passed to parent DeconvolutionGaussianMixtureDataset
        """
        super().__init__(**kwargs)
        
        # Load MNIST dataset
        self.mnist_dataset = DiffMNIST()
        
        # Group MNIST data by digit labels (0-9) to align with mixture components
        self.mnist_by_digit = {}
        for digit in range(10):
            digit_indices = (self.mnist_dataset.labels == digit).nonzero(as_tuple=True)[0]
            self.mnist_by_digit[digit] = digit_indices
        
        # Pre-compute MNIST selections for each group based on component assignments
        self._pre_compute_mnist_selections()
    
    def _pre_compute_mnist_selections(self):
        """Pre-compute MNIST image selections based on mixture component assignments"""
        self.mnist_selections = torch.zeros(self.size, self.group_size, dtype=torch.long)
        
        # Vectorized MNIST selection
        # Flatten all component assignments and convert to digits
        all_component_assignments = self.group_component_info.flatten()  # [size * group_size]
        all_digits = all_component_assignments % 10
        
        # Pre-allocate result tensor
        all_selections = torch.zeros_like(all_component_assignments)
        
        # Process each digit (0-9) in batch
        for digit in range(10):
            # Find all positions that need this digit
            digit_mask = (all_digits == digit)
            n_needed = digit_mask.sum().item()
            
            if n_needed > 0:
                # Get indices for this digit
                digit_indices = self.mnist_by_digit[digit]
                
                # Randomly select n_needed indices (with replacement)
                selected_indices = digit_indices[torch.randint(0, len(digit_indices), (n_needed,))]
                
                # Assign to the appropriate positions
                all_selections[digit_mask] = selected_indices
        
        # Reshape back to [size, group_size]
        self.mnist_selections = all_selections.reshape(self.size, self.group_size)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        # Get the base Gaussian mixture data
        base_data = super().__getitem__(idx)
        
        # Get MNIST images for this group
        mnist_indices = self.mnist_selections[idx]  # [group_size]
        mnist_images = torch.stack([
            self.mnist_dataset.data[idx] for idx in mnist_indices
        ])  # [group_size, 1, 28, 28]
        
        # Structure the data with nested dictionaries
        return {
            'x_0': base_data['x_0'],        # [group_size, data_dim] 
            'x_1': base_data['x_1'],        # [group_size, data_dim]
            'img': mnist_images.float(),      # [group_size, 1, 28, 28]
            'X_0': base_data['X_0']              # [data_dim] - sum of Gaussian counts only
        }

class DeconvolutionMNISTGaussianMultimodalMixtureDataset(DeconvolutionGaussianMixtureDataset):
    """
    Deconvolution dataset that combines MNIST images with Gaussian mixture data.
    
    This dataset extends DeconvolutionGaussianMixtureDataset by:
    1. Using MNIST images selected based on mixture component assignments
    2. Providing structured x_0/x_1 with both 'img' (MNIST) and 'counts' (Gaussian) components
    3. Maintaining the same group-based structure for deconvolution
    
    Each item returns:
      x_0: Dict with 'img' and 'counts' components [group_size, ...]
      x_1: Dict with 'img' (noise) and 'counts' (Gaussian) [group_size, ...]
      z: One-hot matrix indicating component for each group member [group_size, k]
      X_0: Sum of Gaussian counts [data_dim]
    """
    
    def __init__(
        self,
        **kwargs
    ):
        """
        Args:
            **kwargs: Arguments passed to parent DeconvolutionGaussianMixtureDataset
        """
        super().__init__(**kwargs)
        
        # Load MNIST dataset
        self.mnist_dataset = DiffMNIST()
        
        # Group MNIST data by digit labels (0-9) to align with mixture components
        self.mnist_by_digit = {}
        for digit in range(10):
            digit_indices = (self.mnist_dataset.labels == digit).nonzero(as_tuple=True)[0]
            self.mnist_by_digit[digit] = digit_indices
        
        # Pre-compute MNIST selections for each group based on component assignments
        self._pre_compute_mnist_selections()
    
    def _pre_compute_mnist_selections(self):
        """Pre-compute MNIST image selections based on mixture component assignments"""
        self.mnist_selections = torch.zeros(self.size, self.group_size, dtype=torch.long)
        
        # Vectorized MNIST selection
        # Flatten all component assignments and convert to digits
        all_component_assignments = self.group_component_info.flatten()  # [size * group_size]
        all_digits = all_component_assignments % 10
        
        # Pre-allocate result tensor
        all_selections = torch.zeros_like(all_component_assignments)
        
        # Process each digit (0-9) in batch
        for digit in range(10):
            # Find all positions that need this digit
            digit_mask = (all_digits == digit)
            n_needed = digit_mask.sum().item()
            
            if n_needed > 0:
                # Get indices for this digit
                digit_indices = self.mnist_by_digit[digit]
                
                # Randomly select n_needed indices (with replacement)
                selected_indices = digit_indices[torch.randint(0, len(digit_indices), (n_needed,))]
                
                # Assign to the appropriate positions
                all_selections[digit_mask] = selected_indices
        
        # Reshape back to [size, group_size]
        self.mnist_selections = all_selections.reshape(self.size, self.group_size)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        # Get the base Gaussian mixture data
        base_data = super().__getitem__(idx)
        
        # Get MNIST images for this group
        mnist_indices = self.mnist_selections[idx]  # [group_size]
        mnist_images = torch.stack([
            self.mnist_dataset.data[idx] for idx in mnist_indices
        ])  # [group_size, 1, 28, 28]
        
        # Generate noise for x_1 MNIST component (same as DiffMNIST)
        mnist_noise = torch.randn_like(mnist_images)  # [group_size, 1, 28, 28]
        
        # Structure the data with nested dictionaries
        return {
            'x_0': {
                'img': mnist_images.float(),      # [group_size, 1, 28, 28]
                'counts': base_data['x_0']        # [group_size, data_dim] 
            },
            'x_1': {
                'img': mnist_noise.float(),       # [group_size, 1, 28, 28]
                'counts': base_data['x_1']        # [group_size, data_dim]
            },
            'X_0': {'counts': base_data['X_0']}               # [data_dim] - sum of Gaussian counts only
        }
