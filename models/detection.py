from typing import Any
import sys
import types

def build_detection_model(pretrained: bool = True) -> Any:
    """
    Build a DocTR detection predictor with DB + MobileNetV3 Small backbone.
    Returns the predictor (inference module). Training routines can wrap this.
    """
    try:
        # Stub minimal weasyprint to avoid native deps on Windows when importing doctr.io
        if 'weasyprint' not in sys.modules:
            wp = types.ModuleType('weasyprint')
            class _DummyHTML:
                def __init__(self, *args, **kwargs):
                    pass
            wp.HTML = _DummyHTML
            sys.modules['weasyprint'] = wp
        from doctr.models import detection_predictor
        # Try MobileNetV3 Large first, then Small (no ResNet fallback per user request)
        for arch in ("db_mobilenet_v3_large", "db_mobilenet_v3_small"):
            try:
                model = detection_predictor(arch=arch, pretrained=pretrained)
                return model
            except Exception:
                continue
        raise RuntimeError("No compatible DocTR DB MobileNetV3 architecture found (tried large/small)")
    except Exception as e:
        raise RuntimeError(f"Failed to build detection model: {e}")
