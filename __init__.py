from .DifferentialMerger import DifferentialImageMerger, MultiDifferentialMerger

NODE_CLASS_MAPPINGS = {
    "DifferentialImageMerger": DifferentialImageMerger,
    "MultiDifferentialMerger": MultiDifferentialMerger
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DifferentialImageMerger": "Differential Image Merger",
    "MultiDifferentialMerger": "Advanced Differential Merger"
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']