import snapatac2_scooby as snap
from pyfaidx import Fasta
import pyranges as pr
from scipy.sparse import csr_matrix, coo_matrix
from typing import Union, List, Tuple

from dataclasses import dataclass
from typing import List, Union, Optional

import torch
import numpy as np
from scipy import sparse
import pandas as pd
import pickle as pkl

def _build_dna_lut():
    lut = np.full(256, 4, dtype=np.uint8)  # default -> 'N' (4)
    # canonical
    lut[ord('A')] = 0; lut[ord('a')] = 0
    lut[ord('C')] = 1; lut[ord('c')] = 1
    lut[ord('G')] = 2; lut[ord('g')] = 2
    # treat U as T
    lut[ord('T')] = 3; lut[ord('t')] = 3
    lut[ord('U')] = 3; lut[ord('u')] = 3
    # N stays 4
    lut[ord('N')] = 4; lut[ord('n')] = 4
    # (All other IUPAC ambiguity codes R,Y,S,W,K,M,B,D,H,V, gaps, etc. remain 4)
    return lut

_DNA_LUT = _build_dna_lut()

def seq_to_tensor(seq: str) -> np.ndarray:
    """Map sequence to 0:A,1:C,2:G,3:T/U,4:other. Always returns values in [0..4]."""
    arr = np.frombuffer(seq.encode('ascii', 'ignore'), dtype=np.uint8)
    return _DNA_LUT[arr].astype(np.int64, copy=False)

class ExprBySeq:
    """
    Ultra-efficient vectorized version of scExprBySeq that completely avoids Python loops.
    
    This version uses the most efficient approach:
    1. Direct sparse matrix operations without any loops
    2. Vectorized range creation using advanced numpy techniques
    3. Optimal memory usage by keeping everything sparse
    """
    
    def __init__(
        self, 
        snapfile =  "/orcd/data/omarabu/001/gokul/count_sequence/training_data/onek1k_training_data/snapatac_merged_minus3.h5ad",
        fastafile = "/orcd/data/omarabu/001/gokul/count_sequence/training_data/scooby_training_data/genome_human.fa",
        gtffile = "/orcd/data/omarabu/001/gokul/counting_flows/data/gencode.v43.annotation.gtf",
        eligible_genes_path = "/orcd/data/omarabu/001/gokul/count_sequence/training_data/onek1k_training_data/eligible_genes_nnz_by_endpoints.parquet",
        cell_type_col = 'cell_label',
        individual_col = 'individual',
        base_individual = '650_651',
        window_size = 196608,
        batch_size = 100,
        min_value = 0,
        seed = 42,
        train_split = 0.95,
        hvg_only = True,
    ):
        self.data_dim = 196608
        self.window_size = 196608
        self.target_size = 896

        self.annotations = pr.read_gtf(gtffile)
        self.genes = self.annotations[self.annotations.Feature == 'gene'].as_df()

        self.sd = snap.read(snapfile)
        self.fasta = Fasta(fastafile)
        self.chrom_lens = np.array(self.sd.uns['reference_sequences']['reference_seq_length'])
        self.chrom_names = self.sd.uns['reference_sequences']['reference_seq_name']
        self.chrom_starts = np.hstack(([0], self.chrom_lens.cumsum()[:-1]))
        self.chrom_map = {name: i for i, name in enumerate(self.chrom_names)}



        self.chrom_weights = (self.chrom_lens - self.window_size) / (self.chrom_lens - self.window_size).sum()

        self.half_ctx = (self.window_size // 2)  # 196,608//2 = 98,304

        # Build an eligible list of genes that can host the full context window
        chrom_idx = self.genes.Chromosome.map(
            lambda c: self.chrom_map.get(c)
        )
        mask_valid_chr = chrom_idx.notna()
        chrom_idx = chrom_idx.fillna(-1).astype(int)

        # Gene coords are local to the chromosome
        g_start = self.genes.Start.astype(int).to_numpy()
        g_end   = self.genes.End.astype(int).to_numpy()
        cidx    = chrom_idx.to_numpy()

        # Require: gene lies entirely in [half_ctx, chrom_len - half_ctx]
        ok = (cidx >= 0)
        ok &= (g_start >= self.half_ctx)
        ok &= (g_end   <= (self.chrom_lens[cidx] - self.half_ctx))


        elig = pd.DataFrame({
            "chrom_idx": cidx[ok],
            "g_start":   g_start[ok],
            "g_end":     g_end[ok],
        })

        nnz = pd.read_parquet(eligible_genes_path)        
        nnz = nnz[nnz["nnz_endpoint_pad"]]

        both = nnz.merge(elig, on=["chrom_idx","g_start","g_end"], how="inner")

        self._eligible_cidx  = both["chrom_idx"].to_numpy(np.int64)
        self._eligible_gene_name = both["gene_name"]
        self._eligible_start = both["g_start"].to_numpy(np.int64)
        self._eligible_end   = both["g_end"].to_numpy(np.int64)


        if hvg_only:
            hvg_gene_names = pkl.load(open("/orcd/data/omarabu/001/njwfish/counting_flows/results/expr/hvg_gene_names.pkl", "rb"))
            hvg_idx = self._eligible_gene_name.isin(hvg_gene_names)
            self._eligible_cidx = self._eligible_cidx[hvg_idx]
            self._eligible_gene_name = self._eligible_gene_name[hvg_idx]
            self._eligible_start = self._eligible_start[hvg_idx]
            self._eligible_end = self._eligible_end[hvg_idx]
        
        # Cache the full sparse matrix reference
        self.fragment_matrix = self.sd.obsm['fragment_single']
        self.indptr = self.fragment_matrix.indptr
        self.indices = self.fragment_matrix.indices
        self.data = self.fragment_matrix.data

        self.max_nt = self.chrom_lens[-1] + self.chrom_starts[-1]
        self.n_cells = self.sd.n_obs
        self.batch_size = batch_size
        
        self.cell_type_col = cell_type_col
        self.individual_col = individual_col
        self.base_individual = base_individual
        self.base_individual_idx = np.where(self.sd.obs[individual_col] == base_individual)[0]

        # split 10% of non-base individuals into test and exclude from individual idxs
        unique_individuals = self.sd.obs[individual_col].unique()
        unique_individuals = [individual for individual in unique_individuals if individual != base_individual]
        # random permute unique individuals
        rng = np.random.default_rng(seed=seed)
        unique_individuals = rng.permutation(unique_individuals)
        split_idx = int(len(unique_individuals) * train_split)
        print(f"Train split: {split_idx}, Test split: {len(unique_individuals) - split_idx}")
        self.train_individuals, self.test_individuals = unique_individuals[:split_idx], unique_individuals[split_idx:]

        self.individual_idxs = {individual: np.where(self.sd.obs[individual_col] == individual)[0] for individual in self.train_individuals}
        self.test_individual_idxs = {individual: np.where(self.sd.obs[individual_col] == individual)[0] for individual in self.test_individuals}
        print(f"Train cells: {sum(len(idx) for idx in self.individual_idxs.values())}, Test cells: {sum(len(idx) for idx in self.test_individual_idxs.values())}")

        # create one hot encoding for cell type
        target_cond = self.sd.obs[self.cell_type_col].to_numpy()
        self.target_cond = torch.zeros(len(target_cond), len(self.sd.obs[self.cell_type_col].unique()))
        for i, cell_type in enumerate(self.sd.obs[self.cell_type_col].unique()):
            self.target_cond[target_cond == cell_type, i] = 1

    def _global_to_chrom(self, ind):
        chrom_idx = (self.chrom_starts <= ind).nonzero()[0][-1]
        chrom_name = self.chrom_names[int(chrom_idx)]
        local_idx = ind - self.chrom_starts[chrom_idx]
        return chrom_name, int(local_idx)

    def get_seq(self, start: int, end: int) -> np.ndarray:
        # Map start to chrom and stay in that chrom's frame
        # (avoid ever calling _global_to_chrom on `end`).
        chrom_idx = int(np.searchsorted(self.chrom_starts, start, side="right") - 1)
        chrom_name = self.chrom_names[chrom_idx]
        local_start = int(start - self.chrom_starts[chrom_idx])

        # compute local_end purely by span, clamp to chrom end defensively
        span = int(end - start)                     # expected length
        chrom_len = int(self.chrom_lens[chrom_idx])
        local_end = min(local_start + span, chrom_len)

        # fetch sequence
        seq_str = self.fasta[chrom_name][local_start:local_end].seq

        # enforce exact length (trim if a backend returns 1 too many at boundaries)
        if len(seq_str) != span:
            # final safeguard: trim or pad (pad shouldn't be needed if upstream is correct)
            if len(seq_str) > span:
                print(f"WARNING: seq_str length {len(seq_str)} is greater than span {span}")
                seq_str = seq_str[:span]
            else:
                print(f"WARNING: seq_str length {len(seq_str)} is less than span {span}")
                # very unlikely; only if someone sampled past chrom end
                seq_str = seq_str + ("N" * (span - len(seq_str)))

        return seq_to_tensor(seq_str)

    def __len__(self):
        return (self.n_cells // self.batch_size) // 10

    def fast_get_overlap_raw(self, cell_idx=None, window=None):
        """
        Build coverage for all rows in `cell_idx` over [start, end),
        counting ANY fragment that overlaps the window. Duplicates preserved.

        Overlap condition for left-extending runs (val <= 0, interval [col+val, col)):
            max(col+val, start) < min(col, end)
        """

        if window is None:
            window = (0, self.max_nt)
        start, end = window
        length = end - start
        R = len(cell_idx)
        if R == 0 or length <= 0:
            return np.zeros((R, max(0, length)), dtype=np.int32)

        # Gather rows' entry ranges
        row_starts = self.indptr[cell_idx] if cell_idx is not None else self.indptr
        row_ends   = self.indptr[cell_idx + 1] if cell_idx is not None else self.indptr[1:]
        nnz_rows   = row_ends - row_starts
        total = int(nnz_rows.sum())
        if total == 0:
            return np.zeros((R, length), dtype=np.int32)

        # Preallocate and copy slices (tiny loop over rows; avoids large Python concat overhead)
        rr   = np.empty(total, dtype=np.int64)           # row id per entry
        cols = np.empty(total, dtype=np.int64)
        vals = np.empty(total, dtype=np.int64)

        off = 0
        for i, (s, e) in enumerate(zip(row_starts, row_ends)):
            k = e - s
            if k == 0: 
                continue
            rr[off:off+k]   = i
            cols[off:off+k] = self.indices[s:e]
            vals[off:off+k] = self.data[s:e]
            off += k

        # Intervals in global coords: [a, b) with a = col + val (val <= 0), b = col
        a_global = cols + vals
        b_global = cols

        # Fast prefilter for overlap with [start, end):
        # need b > start and a < end
        keep = (b_global > start) & (a_global < end)
        if not np.any(keep):
            return np.zeros((R, length), dtype=np.int32)

        rr   = rr[keep]
        a_g  = a_global[keep]
        b_g  = b_global[keep]

        # Convert to window-local coords and clamp to window
        s = np.maximum(a_g - start, 0)        # start within [0, length]
        e = np.minimum(b_g - start, length)   # end within [0, length]
        keep2 = s < e
        if not np.any(keep2):
            return np.zeros((R, length), dtype=np.int32)
        rr = rr[keep2]
        s  = s[keep2].astype(np.int64, copy=False)
        e  = e[keep2].astype(np.int64, copy=False)
        # Difference-array accumulation via flattened bincount
        Lp1 = length + 1
        flat_start = rr * Lp1 + s
        flat_end   = rr * Lp1 + e
        idx = np.concatenate([flat_start, flat_end])
        wts = np.concatenate([
            np.ones_like(flat_start, dtype=np.int64),
            -np.ones_like(flat_end,   dtype=np.int64)
        ])
        markers = np.bincount(idx, weights=wts, minlength=R * Lp1).reshape(R, Lp1)
        cov = np.cumsum(markers[:, :length], axis=1, dtype=np.int64).astype(np.int32)
        # Return shape like your original (list of vectors) if desired
        return cov

    def sample_gene_uniform_window(self):
        """
        Sample a center uniformly within a (eligible) gene, then return:
        (context_start, context_end, target_start, target_end).

        - Eligibility ensures the full 196,608bp context fits on the chromosome
        for ANY center inside the gene.
        - If length_weighted=True, picks genes ∝ gene length (uniform over all bases across genes).
        Else uniform over genes.
        """
        n = self._eligible_cidx.size
        if n == 0:
            raise RuntimeError("No genes can host the full context window; relax constraints or check reference/GTF.")

        idx = np.random.randint(0, n)

        cidx  = self._eligible_cidx[idx]
        gs, ge = int(self._eligible_start[idx]), int(self._eligible_end[idx])

        # Pick a center uniformly in [gs, ge)
        center_local = int(np.random.randint(gs, ge))
        center_global = int(self.chrom_starts[cidx] + center_local)

        # Build windows
        t_half = self.target_size // 2   # 896 -> 448
        target_start = center_global - t_half
        target_end   = target_start + self.target_size

        ctx_start = center_global - self.half_ctx
        ctx_end   = ctx_start + self.window_size  # = 196,608

        # By construction, these already fit the chromosome (due to eligibility)
        return ctx_start, ctx_end, target_start, target_end

    def __getitem__(self, idx):
        # random start position in genome
        if np.random.random() < 0.9:
            ctx_start, ctx_end, target_start, target_end = self.sample_gene_uniform_window()
        else:
            chrom_idx = np.random.choice(len(self.chrom_starts), p=self.chrom_weights)
            chrom_start = self.chrom_starts[chrom_idx]
            chrom_end = chrom_start + self.chrom_lens[chrom_idx]
            ctx_start = np.random.randint(chrom_start, chrom_end - self.window_size)
            ctx_end = ctx_start + self.window_size
            target_start = (ctx_start + ctx_end) // 2 - self.target_size // 2
            target_end = target_start + self.target_size

        seq = self.get_seq(ctx_start, ctx_end)
        
        # get random set of base cells
        base_cell_idxs = np.random.choice(self.base_individual_idx, size=self.batch_size, replace=True)
        # base_counts = self.fragment_matrix[base_cell_idxs][:, target_start:target_end].toarray()
        base_counts = self.fast_get_overlap_raw(base_cell_idxs, (target_start, target_end))

        # sample random other cell type
        target_individual = np.random.choice(list(self.individual_idxs.keys()))
        target_cell_idxs = np.random.choice(self.individual_idxs[target_individual], size=self.batch_size, replace=True)
        # target_counts = self.fragment_matrix[target_cell_idxs][:, target_start:target_end].toarray()
        target_counts = self.fast_get_overlap_raw(target_cell_idxs, (target_start, target_end))

        return {
            'x_0': target_counts,
            'x_1': base_counts,
            'seq': seq,
            'class_emb': self.target_cond[target_cell_idxs]
        }

