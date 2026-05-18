import sys
from unittest.mock import MagicMock, patch
import pytest

# Mock modules to prevent heavy library loads or device check errors in CI
sys.modules['anomalib.engine'] = MagicMock()
sys.modules['anomalib.models'] = MagicMock()
sys.modules['anomalib.data'] = MagicMock()

with patch('anomalib.utils.path.create_versioned_dir') as mock_path_creator:
    from train import patched_create_versioned_dir, train

def test_patched_create_versioned_dir(tmp_path):
    # Verify the monkey patch avoids WinError 1314 by skip symlink
    root_dir = tmp_path / "results"
    
    # Run patch function
    version_dir = patched_create_versioned_dir(root_dir)
    
    assert version_dir.exists()
    assert version_dir.name == "v0"
    
    # Run second time
    version_dir2 = patched_create_versioned_dir(root_dir)
    assert version_dir2.name == "v1"

@patch('train.Folder')
@patch('train.Patchcore')
@patch('train.Engine')
def test_train_pipeline(mock_engine_cls, mock_patchcore_cls, mock_folder_cls):
    # Setup mock instances
    mock_engine = MagicMock()
    mock_engine_cls.return_value = mock_engine
    
    # Execute train pipeline
    train()
    
    # Assertions
    mock_folder_cls.assert_called_once()
    mock_patchcore_cls.assert_called_once()
    mock_engine_cls.assert_called_once()
    mock_engine.fit.assert_called_once()
