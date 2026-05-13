import torch
import gc

DEFAULT_DTYPE = torch.float32

def cuda_matrix_mult(features: torch.Tensor, W: torch.Tensor, 
                     safety_margin_mb: int = 2048, *, clear_cache: bool = False):
    """
    Multiply a feature matrix by a projection matrix with VRAM-aware chunking.

    This function computes

    ``features @ W``

    while automatically choosing the largest safe row-wise chunk size based on
    currently available CUDA memory. It is intended for large feature matrices
    where the full matrix multiplication may not fit in GPU memory.

    If ``features`` is already on a CUDA device, chunks are processed directly
    on that device. If ``features`` is on CPU, chunks are moved to the selected
    device for multiplication and copied back to the output tensor. The
    function tries to minimize the number of chunks for speed while preserving
    a configurable safety margin in free VRAM.

    When the input and output feature dimensions are equal, the result is
    written back into ``features`` in-place. Otherwise, a new output tensor is
    allocated on the same device as ``features``.

    Parameters
    ----------
    features :
        Input tensor of shape ``(num_rows, in_features)``.
    W :
        Projection or weight matrix of shape ``(in_features, out_features)``.
        The matrix is moved to ``features.device`` before multiplication.
    safety_margin_mb :
        Amount of CUDA memory, in megabytes, to keep unused when estimating the
        maximum chunk size. This reduces the risk of out-of-memory errors from
        temporary allocations or allocator fragmentation.
    clear_cache :
        If ``True`` and running on CUDA, run garbage collection and clear the
        CUDA cache before estimating available memory.

    Returns
    -------
    torch.Tensor
        Result of ``features @ W`` with shape ``(num_rows, out_features)``.
        If ``in_features == out_features``, the returned tensor is the original
        ``features`` tensor modified in-place. Otherwise, a newly allocated
        tensor is returned.

    Raises
    ------
    RuntimeError
        If the available CUDA memory is smaller than the requested safety
        margin, leaving no usable workspace for multiplication.

    Notes
    -----
    The chunk-size estimate assumes that the dominant extra memory cost is the
    output chunk. If the input tensor is on CPU, the estimate also accounts for
    the input chunk copied to CUDA. The estimate is intentionally simple and may
    still fail in extremely fragmented memory conditions or when other kernels
    allocate additional temporary buffers.
    """

    # Ensure W is on the given device
    device = features.device
    W = W.to(device=device)
        
    num_rows, in_features = features.shape
    out_features = W.shape[1]
    element_size = features.element_size()

    if device.type == 'cuda':
        if clear_cache:
            gc.collect()
            torch.cuda.empty_cache()
        
        free_mem, _ = torch.cuda.mem_get_info(device)
        usable_mem = free_mem - (safety_margin_mb * 1024 * 1024)
        
        if usable_mem <= 0:
            raise RuntimeError("Not enough GPU memory, even for the workspace. Free up VRAM or increase chunking.")
            
        # If features is on CPU, we need VRAM for BOTH the input chunk and the output chunk.
        # If features is already on GPU, the input chunk is just a view (0 extra VRAM), 
        # so we only need space for the new output chunk.
        if features.device.type == 'cpu':
            bytes_per_row = (in_features + out_features) * element_size
        else:
            bytes_per_row = out_features * element_size
            
        # Calculate the absolute maximum rows we can process in one go
        chunk_size = max(1, usable_mem // bytes_per_row)
    else:
        # Fallback if running on CPU
        chunk_size = num_rows

    # Cap chunk_size so we don't do unnecessary math
    chunk_size = min(chunk_size, num_rows) 
    
    # We can do in-place replacement ONLY if input and output feature sizes match
    can_overwrite = (in_features == out_features)
    
    if not can_overwrite:
        # If dimensions change, pre-allocate the output tensor on the original device
        out_features_tensor = torch.empty((num_rows, out_features), dtype=features.dtype, device=features.device)

    # Process in mathematically optimal batches
    for start_idx in range(0, num_rows, chunk_size):
        end_idx = min(start_idx + chunk_size, num_rows)
        
        # 1. Extract chunk and move to GPU (if not already there)
        features_chunk = features[start_idx:end_idx]
        if features_chunk.device != device:
            features_chunk = features_chunk.to(device)
        
        # 2. Compute matrix multiplication
        result_chunk = features_chunk @ W
        
        # 3. Store the result and free the chunk memory
        if can_overwrite:
            features[start_idx:end_idx] = result_chunk.to(features.device)
        else:
            out_features_tensor[start_idx:end_idx] = result_chunk.to(features.device)
            
        # Delete intermediate tensors to free VRAM for the next loop
        del features_chunk, result_chunk
            
    return features if can_overwrite else out_features_tensor

def clean_cuda_memory():
    """
    Release cached CUDA memory and synchronize the CUDA device.

    This helper performs a conservative cleanup step after memory-heavy
    operations. If CUDA is available, it clears unused cached memory, waits for
    queued CUDA work to finish, and runs Python garbage collection.

    The function is safe to call on CPU-only systems; in that case it performs
    no CUDA operations.

    Notes
    -----
    ``torch.cuda.empty_cache()`` does not free memory occupied by live tensors.
    It only releases unused cached blocks held by PyTorch's CUDA allocator.
    Therefore, tensors that are no longer needed should be deleted before
    calling this function.
    """
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        gc.collect()