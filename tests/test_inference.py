import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from PIL import Image

# Mock modules to prevent heavy library loads or device check errors in CI
sys.modules["anomalib.deploy.inferencers.torch_inferencer"] = MagicMock()

# Import the target engine
from inference_engine import IntegratedEngine


@pytest.fixture
def mock_engine():
    # Setup patches to prevent TorchInferencer and SAM2 build from actually running
    with (
        patch("inference_engine.TorchInferencer") as mock_torch_inf,
        patch("inference_engine.build_sam2"),
        patch("inference_engine.SAM2ImagePredictor") as mock_sam_pred,
        patch("inference_engine.SAM2_AVAILABLE", True),
    ):
        # Configure mocks
        mock_inf_instance = MagicMock()
        mock_torch_inf.return_value = mock_inf_instance

        # Configure model prediction shape
        mock_prediction = MagicMock()
        mock_prediction.anomaly_map = np.random.rand(256, 256).astype(np.float32)
        mock_prediction.pred_score = 0.85
        mock_inf_instance.model.return_value = mock_prediction

        # Configure SAM2 predictor mock
        mock_pred_instance = MagicMock()
        mock_sam_pred.return_value = mock_pred_instance
        mock_pred_instance.predict.return_value = (
            [np.zeros((100, 100), dtype=bool)],
            [0.99],
            None,
        )

        engine = IntegratedEngine(anomalib_path="dummy_path.pt", sam2_checkpoint="dummy_sam_tiny.pt")
        yield engine


def test_device_detection(mock_engine):
    device = mock_engine._get_device()
    assert device in ["cuda", "mps", "cpu"]


def test_analyze_anomalib(mock_engine):
    # Create dummy image
    img = Image.fromarray(np.uint8(np.random.rand(100, 100, 3) * 255))

    # Create dummy prediction mock to avoid format string errors
    dummy_pred = {
        "anomaly_map": torch.zeros((1, 1, 256, 256)),
        "pred_score": torch.tensor([0.99])
    }
    
    # Run analysis with patched model call
    mock_engine.anomalib_engine.model.model = MagicMock(return_value=dummy_pred)
    res = mock_engine.analyze_anomalib(img)

    # Assertions
    assert "heatmap" in res
    assert "score" in res
    assert "peak_point" in res
    assert isinstance(res["score"], float)
    assert len(res["peak_point"]) == 2


def test_segment_with_sam2(mock_engine):
    img = Image.fromarray(np.uint8(np.random.rand(100, 100, 3) * 255))
    points = np.array([[50, 50]], dtype=np.float32)
    labels = np.array([1], dtype=np.int32)

    mask = mock_engine.segment_with_sam2(img, points, labels)

    assert mask is not None
    assert mask.shape == (100, 100)


def test_create_overlay(mock_engine):
    img = Image.fromarray(np.zeros((100, 100, 3), dtype=np.uint8))
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[40:60, 40:60] = 1

    overlaid = mock_engine.create_overlay(img, mask)
    assert isinstance(overlaid, Image.Image)
    assert overlaid.size == (100, 100)


def test_create_heatmap_overlay(mock_engine):
    img = Image.fromarray(np.zeros((100, 100, 3), dtype=np.uint8))
    heatmap = np.random.rand(100, 100).astype(np.float32)

    overlaid = mock_engine.create_heatmap_overlay(img, heatmap)
    assert isinstance(overlaid, Image.Image)
    assert overlaid.size == (100, 100)
