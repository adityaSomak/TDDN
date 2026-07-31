"""Cityscapes 19-class semantic segmentation class names.

Verbatim spellings and order from mmsegmentation's ``CityscapesDataset``
(the standard 19 "trainId" evaluation classes). No "background" class in
this taxonomy at all -- Cityscapes' ignore/void pixels are a separate
label, not a semantic class, so no exclusion logic is needed here (unlike
PASCAL VOC/Context, which do have a real catch-all "background" class).
"""

CITYSCAPES_CLASSES = [
    "road", "sidewalk", "building", "wall", "fence", "pole", "traffic light",
    "traffic sign", "vegetation", "terrain", "sky", "person", "rider", "car",
    "truck", "bus", "train", "motorcycle", "bicycle",
]

assert len(CITYSCAPES_CLASSES) == 19, f"Expected 19 classes, got {len(CITYSCAPES_CLASSES)}"
