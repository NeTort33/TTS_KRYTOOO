import torch
import pickle
import numpy as np
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class ModelLoader:
    """Loader for custom voice models (.pth and .index files)"""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"ModelLoader initialized on device: {self.device}")
    
    def load_pth_model(self, pth_path):
        """
        Load PyTorch model from .pth file
        
        Args:
            pth_path: Path to .pth file
            
        Returns:
            Loaded model or None if failed
        """
        try:
            logger.info(f"Loading PyTorch model from: {pth_path}")
            
            # Load the model checkpoint
            checkpoint = torch.load(pth_path, map_location=self.device)
            
            # Extract model information
            model_info = {}
            
            if isinstance(checkpoint, dict):
                # Standard checkpoint format
                if 'model' in checkpoint:
                    model = checkpoint['model']
                elif 'state_dict' in checkpoint:
                    model = checkpoint['state_dict']
                elif 'net_g' in checkpoint:
                    # Common in voice conversion models
                    model = checkpoint['net_g']
                else:
                    # Assume the checkpoint itself is the model
                    model = checkpoint
                
                # Extract metadata if available
                if 'epoch' in checkpoint:
                    model_info['epoch'] = checkpoint['epoch']
                if 'step' in checkpoint:
                    model_info['step'] = checkpoint['step']
                if 'lr' in checkpoint:
                    model_info['learning_rate'] = checkpoint['lr']
                if 'version' in checkpoint:
                    model_info['version'] = checkpoint['version']
                    
            else:
                # Model weights only
                model = checkpoint
            
            # Handle different model formats
            if isinstance(model, dict):
                # Check if this is a state dict by looking for tensor values
                has_tensors = any(isinstance(v, torch.Tensor) for v in model.values() if v is not None)
                has_weight_keys = any('weight' in k or 'bias' in k for k in model.keys())
                
                if has_tensors and has_weight_keys and all(isinstance(k, str) for k in model.keys()):
                    # This looks like a state dict, create a simple wrapper
                    try:
                        model = SimpleModelWrapper(model)
                        logger.info("Created model wrapper for state dict")
                    except Exception as e:
                        logger.warning(f"Could not create model wrapper: {e}")
                        # Keep the original dict, we'll handle it in TTS engine
                else:
                    logger.info("Model is a dictionary but not a standard state dict")
            
            # Count parameters safely
            try:
                if hasattr(model, 'parameters') and callable(getattr(model, 'parameters')):
                    model_info['parameters'] = sum(p.numel() for p in model.parameters())
                elif isinstance(model, dict):
                    # Count parameters in state dict
                    total_params = 0
                    for v in model.values():
                        if isinstance(v, torch.Tensor):
                            total_params += v.numel()
                    model_info['parameters'] = total_params if total_params > 0 else 'Unknown'
                else:
                    model_info['parameters'] = 'Unknown'
            except Exception as e:
                logger.warning(f"Error counting parameters: {e}")
                model_info['parameters'] = 'Unknown'
            
            model_info['size_mb'] = Path(pth_path).stat().st_size / (1024 * 1024)
            
            logger.info(f"Model loaded successfully. Parameters: {model_info.get('parameters', 'Unknown')}")
            return model, model_info
            
        except Exception as e:
            logger.error(f"Error loading PyTorch model: {str(e)}")
            return None, {}
    
    def load_index_file(self, index_path):
        """
        Load index file (usually contains feature mappings or embeddings)
        
        Args:
            index_path: Path to .index file
            
        Returns:
            Loaded index data or None if failed
        """
        try:
            logger.info(f"Loading index file from: {index_path}")
            
            # Try different formats
            index_data = None
            
            # Try pickle format first
            try:
                with open(index_path, 'rb') as f:
                    index_data = pickle.load(f)
                logger.info("Index file loaded as pickle format")
            except:
                pass
            
            # Try numpy format
            if index_data is None:
                try:
                    index_data = np.load(index_path, allow_pickle=True)
                    logger.info("Index file loaded as numpy format")
                except:
                    pass
            
            # Try torch format
            if index_data is None:
                try:
                    index_data = torch.load(index_path, map_location=self.device)
                    logger.info("Index file loaded as torch format")
                except:
                    pass
            
            # Try text format (for simple mappings)
            if index_data is None:
                try:
                    with open(index_path, 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                    index_data = {}
                    for line in lines:
                        if '=' in line or ':' in line:
                            if '=' in line:
                                key, value = line.strip().split('=', 1)
                            else:
                                key, value = line.strip().split(':', 1)
                            index_data[key.strip()] = value.strip()
                    logger.info("Index file loaded as text format")
                except:
                    pass
            
            if index_data is None:
                logger.warning("Could not load index file in any known format")
                return {}
            
            return index_data
            
        except Exception as e:
            logger.error(f"Error loading index file: {str(e)}")
            return {}
    
    def load_model(self, pth_path, index_path):
        """
        Load complete model with both .pth and .index files
        
        Args:
            pth_path: Path to .pth model file
            index_path: Path to .index file
            
        Returns:
            Dictionary containing model data or None if failed
        """
        try:
            logger.info("Loading complete model...")
            
            # Load PyTorch model
            model, model_info = self.load_pth_model(pth_path)
            if model is None:
                logger.error("Failed to load PyTorch model")
                return None
            
            # Load index data
            index_data = self.load_index_file(index_path)
            
            # Combine everything
            model_data = {
                'model': model,
                'index_data': index_data,
                'model_info': model_info,
                'pth_path': str(pth_path),
                'index_path': str(index_path)
            }
            
            logger.info("Complete model loaded successfully")
            return model_data
            
        except Exception as e:
            logger.error(f"Error loading complete model: {str(e)}")
            return None

class SimpleModelWrapper(torch.nn.Module):
    """Simple wrapper for state dict models"""
    
    def __init__(self, state_dict):
        super().__init__()
        self.state_dict_data = state_dict
        
        # Create simple layers based on state dict
        self.layers = torch.nn.ModuleDict()
        
        for name, tensor in state_dict.items():
            if 'weight' in name:
                layer_name = name.replace('.weight', '').replace('.', '_')
                if tensor.dim() == 2:
                    # Linear layer
                    self.layers[layer_name] = torch.nn.Linear(tensor.size(1), tensor.size(0))
                elif tensor.dim() == 3:
                    # Conv1d layer
                    self.layers[layer_name] = torch.nn.Conv1d(tensor.size(1), tensor.size(0), tensor.size(2))
                elif tensor.dim() == 4:
                    # Conv2d layer  
                    self.layers[layer_name] = torch.nn.Conv2d(tensor.size(1), tensor.size(0), 
                                                            (tensor.size(2), tensor.size(3)))
        
        # Load the actual weights
        self.load_state_dict(state_dict, strict=False)
    
    def forward(self, x):
        """Simple forward pass"""
        # Apply available layers sequentially
        for layer in self.layers.values():
            try:
                if isinstance(layer, torch.nn.Linear) and x.dim() > 2:
                    x = x.view(x.size(0), -1)
                x = layer(x)
                x = torch.relu(x)  # Simple activation
            except:
                continue
        return x
