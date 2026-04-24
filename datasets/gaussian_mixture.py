"""
Simple PyTorch Datasets for Integer-ized Gaussian Mixture Models

Standard PyTorch datasets with mixture of Gaussians, including low-rank variants.
"""

import torch
from torch.utils.data import Dataset
from torch.distributions import MultivariateNormal, Normal, Dirichlet
import numpy as np
from typing import Dict, Any, Tuple, Optional


class GaussianMixtureDataset(Dataset):
    """
    Mixture of Gaussians dataset with integer-ized outputs.
    
    Pre-samples all source and target count vectors at initialization
    for fast training with mixtures of Gaussian distributions.
    Designed to scale easily across different dimensions.
    """
    
    def __init__(
        self,
        size: int,
        data_dim: int,
        value_range: int = 1024,
        min_value: int = 0,
        k: int = 3,
        mean_scale: float = 10.0,
        cov_scale: float = 5.0,
        min_eigenvalue: float = 0.1,
        seed: int = 42,
    ):
        """
        Args:
            size: Dataset size (number of samples)
            data_dim: Dimensionality of vectors
            value_range: Maximum value for integer outputs
            min_value: Minimum value for integer outputs
            k: Number of mixture components
            mean_scale: Scale for sampling mixture component means
            cov_scale: Scale for sampling covariance eigenvalues
            min_eigenvalue: Minimum eigenvalue to ensure positive definiteness
            seed: Random seed for reproducibility
        """
        self.size = size
        self.data_dim = data_dim
        self.value_range = value_range
        self.min_value = min_value
        self.k = k
        self.mean_scale = mean_scale
        self.cov_scale = cov_scale
        self.min_eigenvalue = min_eigenvalue
        self.seed = seed

        print(f"Pre-sampling {size} samples with d={data_dim}, k={k}...")
        
        # Pre-sample all data
        self._pre_sample_all_data()
        
        print(f"✓ Pre-sampling complete!")
    
    def _pre_sample_all_data(self):
        """Pre-sample all source and target data at initialization"""
        # Pre-allocate tensors for all data
        self.x0_data = torch.zeros(self.size, self.data_dim, dtype=torch.int32)
        self.x1_data = torch.zeros(self.size, self.data_dim, dtype=torch.int32)
        
        # Sample mixture parameters
        (self.means_source, self.covs_source, self.weights_source, 
         self.means_target, self.covs_target, self.weights_target) = self._sample_parameters()
        
        # Generate all samples vectorized and track component assignments
        self.x0_data, self.x0_components = self._sample_from_mixture_vectorized(self.means_source, self.covs_source, self.weights_source, self.size)
        self.x1_data, self.x1_components = self._sample_from_mixture_vectorized(self.means_target, self.covs_target, self.weights_target, self.size)
    
    def _sample_parameters(self):
        """Sample new mixture parameters with scalable priors"""
        with torch.random.fork_rng():
            torch.manual_seed(self.seed)
            
            # Sample means: k components, each with d-dimensional mean
            # Scale means proportionally to sqrt(d) to maintain reasonable scale
            mean_std = self.mean_scale / np.sqrt(self.data_dim)
            print("mean_std", mean_std)
            means_source = torch.randn(self.k, self.data_dim) * mean_std + (self.min_value + self.value_range) / 2  
            means_target = torch.randn(self.k, self.data_dim) * mean_std + (self.min_value + self.value_range) / 2  
            
            # Sample covariance matrices using eigenvalue decomposition
            # This ensures positive definiteness and allows scaling
            covs_source = self._sample_covariance_matrices()
            covs_target = self._sample_covariance_matrices()
            
            # Sample mixture weights
            weights_source = Dirichlet(torch.ones(self.k)).sample()
            weights_target = Dirichlet(torch.ones(self.k)).sample()
        
        return means_source, covs_source, weights_source, means_target, covs_target, weights_target
    
    def _sample_covariance_matrices(self):
        """Sample positive definite covariance matrices that scale well with dimension"""
        covs = torch.zeros(self.k, self.data_dim, self.data_dim)
        
        for i in range(self.k):
            # Sample eigenvalues - use exponential distribution to ensure positivity
            # Scale eigenvalues to maintain reasonable condition numbers
            eigenvalues = torch.distributions.Exponential(1.0).sample((self.data_dim,)) * self.cov_scale
            eigenvalues = torch.clamp(eigenvalues, min=self.min_eigenvalue)
            
            # Sample random orthogonal matrix (rotation)
            Q, _ = torch.linalg.qr(torch.randn(self.data_dim, self.data_dim))
            
            # Construct covariance: Q @ diag(eigenvalues) @ Q^T
            Lambda = torch.diag(eigenvalues)
            covs[i] = Q @ Lambda @ Q.T
        
        return covs
    
    def _sample_from_mixture_vectorized(self, means: torch.Tensor, covs: torch.Tensor, 
                                       weights: torch.Tensor, num_samples: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Vectorized sampling from mixture of Gaussians
        
        Args:
            means: Component means [k, d]
            covs: Component covariances [k, d, d]
            weights: Mixture weights [k]
            num_samples: Number of samples to generate
        
        Returns:
            Batch of integer-ized samples [num_samples, data_dim]
            Component assignments [num_samples]
        """
        # Sample which components to use for each sample
        components = torch.multinomial(weights, num_samples, replacement=True)
        
        # Sample from each component
        samples = torch.zeros(num_samples, self.data_dim)
        
        for i in range(self.k):
            # Find samples that should come from component i
            mask = (components == i)
            n_samples_i = mask.sum().item()
            
            if n_samples_i > 0:
                # Sample from multivariate normal
                dist = MultivariateNormal(means[i], covs[i])
                component_samples = dist.sample((n_samples_i,))
                samples[mask] = component_samples
        
        # Integer-ize by rounding and reflection
        samples = torch.round(samples)
        samples = self._reflect_boundaries(samples)
        
        return samples.int(), components
    
    def _reflect_boundaries(self, samples: torch.Tensor) -> torch.Tensor:
        """
        Apply reflection at boundaries to preserve mass instead of clipping.
        
        This method reflects values that fall outside [min_value, max_value] 
        back into the valid range, preserving the total probability mass.
        """
        max_value = self.value_range - 1
        
        # Handle samples below min_value
        below_min = samples < self.min_value
        if below_min.any():
            # Reflect: new_value = min_value + (min_value - old_value)
            samples[below_min] = 2 * self.min_value - samples[below_min]
        
        # Handle samples above max_value  
        above_max = samples > max_value
        if above_max.any():
            # Reflect: new_value = max_value - (old_value - max_value)
            samples[above_max] = 2 * max_value - samples[above_max]
        
        # Apply reflection iteratively until all samples are in bounds
        # (needed for extreme outliers that might reflect multiple times)
        still_out_of_bounds = (samples < self.min_value) | (samples > max_value)
        if still_out_of_bounds.any():
            samples = self._reflect_boundaries(samples)  # Recursive call
            
        return samples
    
    def __len__(self) -> int:
        """Return dataset size"""
        return self.size
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Return pre-computed sample"""
        target_idx = np.random.randint(0, self.size)
        return {
            'x_0': self.x0_data[idx],
            'x_1': self.x1_data[target_idx]
        }


class LowRankGaussianMixtureDataset(Dataset):
    """
    Low-rank mixture of Gaussians dataset with integer-ized outputs.
    
    Samples from k-component Gaussian mixture in r dimensions, then projects
    to d dimensions via linear transformation and adds isotropic noise.
    This allows modeling high-dimensional data with low intrinsic dimensionality.
    """
    
    def __init__(
        self,
        size: int,
        data_dim: int,
        latent_dim: int,
        value_range: int = 1024,
        min_value: int = 0,
        k: int = 3,
        mean_scale: float = 10.0,
        cov_scale: float = 5.0,
        noise_scale: float = 1.0,
        projection_scale: float = 1.0,
        min_eigenvalue: float = 0.1,
        seed: int = 42,
        continuous: bool = False,
        diagonal_cov: bool = False,
    ):
        """
        Args:
            size: Dataset size (number of samples)
            data_dim: Output dimensionality (d)
            latent_dim: Latent dimensionality (r, should be < data_dim)
            value_range: Maximum value for integer outputs
            min_value: Minimum value for integer outputs
            k: Number of mixture components
            mean_scale: Scale for sampling mixture component means in latent space
            cov_scale: Scale for sampling covariance eigenvalues in latent space
            noise_scale: Scale of isotropic noise added after projection
            projection_scale: Scale of random projection matrix
            min_eigenvalue: Minimum eigenvalue to ensure positive definiteness
            seed: Random seed for reproducibility
            continuous: If True, skip integer rounding and boundary reflection so
                samples remain a clean continuous Gaussian mixture in data space
                (the closed-form Bayes denoiser in models/bayes.py is the exact
                Bayes-optimal denoiser for this distribution). Default False
                preserves the original integer count behaviour.
        """
        self.size = size
        self.data_dim = data_dim
        self.latent_dim = latent_dim
        self.value_range = value_range
        self.min_value = min_value
        self.k = k
        self.mean_scale = mean_scale
        self.cov_scale = cov_scale
        self.noise_scale = noise_scale
        self.projection_scale = projection_scale
        self.min_eigenvalue = min_eigenvalue
        self.seed = seed
        self.continuous = continuous

        if latent_dim >= data_dim:
            raise ValueError(f"latent_dim ({latent_dim}) should be < data_dim ({data_dim})")

        print(f"Pre-sampling {size} samples with d={data_dim}, r={latent_dim}, k={k}...")
        
        # Pre-sample all data
        self._pre_sample_all_data()
        
        print(f"✓ Pre-sampling complete!")
    
    def _pre_sample_all_data(self):
        """Pre-sample all source and target data at initialization"""
        # Pre-allocate tensors for all data
        out_dtype = torch.float32 if self.continuous else torch.int32
        self.x0_data = torch.zeros(self.size, self.data_dim, dtype=out_dtype)
        self.x1_data = torch.zeros(self.size, self.data_dim, dtype=out_dtype)

        # Sample mixture parameters and projection matrices
        (self.means_source, self.covs_source, self.weights_source, self.proj_source,
         self.means_target, self.covs_target, self.weights_target, self.proj_target) = self._sample_parameters()

        # Generate all samples vectorized and track component assignments
        self.x0_data, self.x0_components = self._sample_from_low_rank_mixture(
            self.means_source, self.covs_source, self.weights_source, self.proj_source, self.size)
        self.x1_data, self.x1_components = self._sample_from_low_rank_mixture(
            self.means_target, self.covs_target, self.weights_target, self.proj_target, self.size)
    
    def _sample_parameters(self):
        """Sample mixture parameters in latent space and projection matrices"""
        with torch.random.fork_rng():
            torch.manual_seed(self.seed)
            
            # Sample means in latent space
            mean_std = self.mean_scale / np.sqrt(self.latent_dim)
            means_source = torch.randn(self.k, self.latent_dim) * mean_std + (self.min_value + self.value_range) / 2  
            means_target = torch.randn(self.k, self.latent_dim) * mean_std + (self.min_value + self.value_range) / 2  
            
            # Sample covariance matrices in latent space
            covs_source = self._sample_latent_covariance_matrices()
            covs_target = self._sample_latent_covariance_matrices()
            
            # Sample mixture weights
            weights_source = Dirichlet(torch.ones(self.k)).sample()
            weights_target = Dirichlet(torch.ones(self.k)).sample()
            
            # Sample projection matrices: d x r
            proj_scale = self.projection_scale / np.sqrt(self.latent_dim)
            proj_source = torch.randn(self.data_dim, self.latent_dim) * proj_scale
            proj_target = torch.randn(self.data_dim, self.latent_dim) * proj_scale
        
        return (means_source, covs_source, weights_source, proj_source,
                means_target, covs_target, weights_target, proj_target)
    
    def _sample_latent_covariance_matrices(self):
        """Sample covariance matrices in latent space"""
        covs = torch.zeros(self.k, self.latent_dim, self.latent_dim)

        for i in range(self.k):
            # Sample eigenvalues
            eigenvalues = torch.distributions.Exponential(1.0).sample((self.latent_dim,)) * self.cov_scale
            eigenvalues = torch.clamp(eigenvalues, min=self.min_eigenvalue)
            Lambda = torch.diag(eigenvalues)

            # Sample random orthogonal matrix
            Q, _ = torch.linalg.qr(torch.randn(self.latent_dim, self.latent_dim))
            covs[i] = Q @ Lambda @ Q.T

        return covs
    
    def _sample_from_low_rank_mixture(self, means: torch.Tensor, covs: torch.Tensor,
                                     weights: torch.Tensor, projection: torch.Tensor,
                                     num_samples: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample from low-rank Gaussian mixture
        
        Args:
            means: Component means in latent space [k, r]
            covs: Component covariances in latent space [k, r, r]
            weights: Mixture weights [k]
            projection: Projection matrix [d, r]
            num_samples: Number of samples to generate
        
        Returns:
            Batch of integer-ized samples [num_samples, data_dim]
            Component assignments [num_samples]
        """
        # Sample which components to use
        components = torch.multinomial(weights, num_samples, replacement=True)
        
        # Sample in latent space
        latent_samples = torch.zeros(num_samples, self.latent_dim)
        
        for i in range(self.k):
            mask = (components == i)
            n_samples_i = mask.sum().item()
            
            if n_samples_i > 0:
                dist = MultivariateNormal(means[i], covs[i])
                component_samples = dist.sample((n_samples_i,))
                latent_samples[mask] = component_samples
        
        # Project to high-dimensional space
        projected_samples = latent_samples @ projection.T  # [num_samples, d]
        
        # Add isotropic noise
        noise = torch.randn_like(projected_samples) * self.noise_scale
        samples = projected_samples + noise

        if self.continuous:
            # Skip rounding/reflection: clean continuous Gaussian mixture in data space.
            return samples.float(), components

        # Integer-ize
        samples = torch.round(samples)
        samples = self._reflect_boundaries(samples)

        return samples.int(), components
    
    def _reflect_boundaries(self, samples: torch.Tensor) -> torch.Tensor:
        """
        Apply reflection at boundaries to preserve mass instead of clipping.
        
        This method reflects values that fall outside [min_value, max_value] 
        back into the valid range, preserving the total probability mass.
        """
        max_value = self.value_range - 1
        
        # Handle samples below min_value
        below_min = samples < self.min_value
        if below_min.any():
            # Reflect: new_value = min_value + (min_value - old_value)
            samples[below_min] = 2 * self.min_value - samples[below_min]
        
        # Handle samples above max_value  
        above_max = samples > max_value
        if above_max.any():
            # Reflect: new_value = max_value - (old_value - max_value)
            samples[above_max] = 2 * max_value - samples[above_max]
        
        # Apply reflection iteratively until all samples are in bounds
        # (needed for extreme outliers that might reflect multiple times)
        still_out_of_bounds = (samples < self.min_value) | (samples > max_value)
        if still_out_of_bounds.any():
            samples = self._reflect_boundaries(samples)  # Recursive call
            
        return samples
    
    def __len__(self) -> int:
        """Return dataset size"""
        return self.size
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Return pre-computed sample"""
        target_idx = np.random.randint(0, self.size)
        return {
            'x_0': self.x0_data[idx],
            'x_1': self.x1_data[target_idx]
        }
