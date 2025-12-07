"""
FAISS index management with HNSW and persistence.
"""
import os
import faiss
import numpy as np
from pathlib import Path
from typing import Optional, List, Tuple


class FAISSIndexManager:
    """Manages FAISS indexes with HNSW and persistence."""
    
    def __init__(self, index_dir: str = "faiss_indexes"):
        """
        Initialize index manager.
        
        Args:
            index_dir: Directory to store persistent indexes
        """
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(exist_ok=True)
    
    def get_index_path(self, index_name: str) -> Path:
        """Get path for index file."""
        return self.index_dir / f"{index_name}.index"
    
    def build_hnsw_index(
        self,
        vectors: np.ndarray,
        index_name: str,
        m: int = 32,
        ef_construction: int = 200
    ) -> faiss.Index:
        """
        Build HNSW index from vectors.
        
        Args:
            vectors: Numpy array of shape (n, dim) with float32 vectors
            index_name: Name for the index (used for persistence)
            m: Number of connections per node (default 32, higher = more accurate but slower)
            ef_construction: Size of dynamic candidate list (default 200)
        
        Returns:
            FAISS HNSW index (or IndexFlatIP fallback if HNSW not available)
        """
        dim = vectors.shape[1]
        n = vectors.shape[0]
        
        if n == 0:
            raise ValueError("Cannot build index with empty vectors")
        
        # Normalize vectors for cosine similarity
        faiss.normalize_L2(vectors)
        
        # Build HNSW index - try different class names for compatibility
        index = None
        try:
            # Try IndexHNSWFlat (most common)
            index = faiss.IndexHNSWFlat(dim, m)
            if hasattr(index, 'hnsw') and hasattr(index.hnsw, 'efConstruction'):
                index.hnsw.efConstruction = ef_construction
        except (AttributeError, TypeError) as e:
            try:
                # Fallback to IndexHNSWFlat32
                index = faiss.IndexHNSWFlat32(dim, m)
                if hasattr(index, 'hnsw') and hasattr(index.hnsw, 'efConstruction'):
                    index.hnsw.efConstruction = ef_construction
            except (AttributeError, TypeError):
                # Last resort: use IndexFlatIP (exact search, no HNSW)
                print(f"Warning: HNSW not available, using IndexFlatIP (exact search). Error: {e}")
                index = faiss.IndexFlatIP(dim)
        
        # Add vectors
        index.add(vectors)
        
        # Save to disk
        self.save_index(index, index_name)
        
        return index
    
    def load_index(self, index_name: str) -> Optional[faiss.Index]:
        """
        Load index from disk.
        
        Args:
            index_name: Name of the index
        
        Returns:
            FAISS index or None if not found
        """
        index_path = self.get_index_path(index_name)
        if not index_path.exists():
            return None
        
        try:
            index = faiss.read_index(str(index_path))
            return index
        except Exception as e:
            print(f"Error loading index {index_name}: {e}")
            return None
    
    def save_index(self, index: faiss.Index, index_name: str) -> bool:
        """
        Save index to disk.
        
        Args:
            index: FAISS index to save
            index_name: Name for the index
        
        Returns:
            True if successful
        """
        try:
            index_path = self.get_index_path(index_name)
            faiss.write_index(index, str(index_path))
            return True
        except Exception as e:
            print(f"Error saving index {index_name}: {e}")
            return False
    
    def search(
        self,
        index: faiss.Index,
        query_vectors: np.ndarray,
        k: int = 10,
        ef_search: int = 50
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search in HNSW index.
        
        Args:
            index: FAISS index
            query_vectors: Query vectors (n, dim)
            k: Number of nearest neighbors to return
            ef_search: Size of dynamic candidate list for search (default 50)
        
        Returns:
            Tuple of (distances, indices)
        """
        # Normalize query vectors (use copy to avoid modifying input)
        query_vectors = query_vectors.copy()
        faiss.normalize_L2(query_vectors)
        
        # Set ef_search for HNSW (if applicable)
        # Check for HNSW index more robustly
        if hasattr(index, 'hnsw') and hasattr(index.hnsw, 'efSearch'):
            index.hnsw.efSearch = ef_search
        
        # Search
        distances, indices = index.search(query_vectors, k)
        
        return distances, indices
    
    def update_index(
        self,
        index_name: str,
        new_vectors: np.ndarray,
        new_ids: Optional[List[int]] = None
    ) -> faiss.Index:
        """
        Update existing index with new vectors.
        
        Args:
            index_name: Name of the index
            new_vectors: New vectors to add
            new_ids: Optional IDs for new vectors
        
        Returns:
            Updated index
        """
        # Load existing index or create new
        index = self.load_index(index_name)
        
        if index is None:
            # Create new index
            return self.build_hnsw_index(new_vectors, index_name)
        
        # Normalize new vectors
        faiss.normalize_L2(new_vectors)
        
        # Add to existing index
        index.add(new_vectors)
        
        # Save updated index
        self.save_index(index, index_name)
        
        return index
    
    def index_exists(self, index_name: str) -> bool:
        """Check if index exists on disk."""
        return self.get_index_path(index_name).exists()
    
    def delete_index(self, index_name: str) -> bool:
        """Delete index from disk."""
        index_path = self.get_index_path(index_name)
        if index_path.exists():
            index_path.unlink()
            return True
        return False

