"""
Configuration Management for Virtual In Silico Virus Laboratory
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, field

@dataclass
class VLabConfig:
    """Configuration for VLab pipeline"""
    
    # Paths
    output_dir: Path = Path("results")
    data_dir: Path = Path("data")
    models_dir: Path = Path("models")
    cache_dir: Path = Path(".cache")
    
    # Computational resources
    use_gpu: bool = True
    gpu_id: int = 0
    num_workers: int = 4
    batch_size: int = 8
    mixed_precision: bool = True
    
    # AlphaFold settings
    alphafold_path: Optional[Path] = None
    use_alphafold: bool = True
    alphafold_model: str = "model_1_multimer_v3"  # or model_2_multimer_v3
    max_recycles: int = 3
    num_samples: int = 1
    
    # Assembly simulation
    assembly_method: str = "hybrid"  # "md", "ml", "hybrid"
    md_steps: int = 1000000
    md_timestep: float = 0.01  # ps
    temperature: float = 310.0  # K
    
    # Viability prediction
    viability_model_path: Optional[Path] = None
    viability_threshold: float = 0.5
    confidence_threshold: float = 0.8
    
    # Host prediction
    host_model_path: Optional[Path] = None
    receptor_db_path: Optional[Path] = None
    
    # Infection simulation
    simulate_infection: bool = True
    simulation_time: float = 24.0  # hours
    time_step: float = 0.1  # hours
    
    # Output settings
    generate_3d_model: bool = True
    generate_report: bool = True
    save_intermediates: bool = False
    visualization_quality: str = "high"  # "low", "medium", "high"
    
    # Performance
    timeout_hours: float = 24.0
    checkpoint_interval: int = 3600  # seconds
    
    @classmethod
    def from_file(cls, config_path: Path) -> "VLabConfig":
        """Load configuration from YAML file"""
        if not config_path.exists():
            # Create default config
            config = cls()
            config.save(config_path)
            return config
        
        with open(config_path, 'r') as f:
            data = yaml.safe_load(f) or {}
        
        # Convert paths
        for key in ['output_dir', 'data_dir', 'models_dir', 'cache_dir', 
                   'alphafold_path', 'viability_model_path', 'host_model_path', 
                   'receptor_db_path']:
            if key in data and data[key]:
                data[key] = Path(data[key])
        
        return cls(**data)
    
    def save(self, config_path: Path):
        """Save configuration to YAML file"""
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {}
        for key, value in self.__dict__.items():
            if isinstance(value, Path):
                data[key] = str(value)
            else:
                data[key] = value
        
        with open(config_path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, Path):
                result[key] = str(value)
            else:
                result[key] = value
        return result

