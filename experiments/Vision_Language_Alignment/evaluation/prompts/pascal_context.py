"""PASCAL Context 59-class semantic segmentation class names ("C59").

Verbatim spellings and order from mmsegmentation's
``PascalContextDataset59`` -- the standard GroupViT/TCL-lineage "without
background" Context variant. The full 60-class taxonomy prepends
"background" at index 0; C59 drops it (same no-bg-prompt convention
already used for VOC-20 -- GT background pixels -> ignore, real classes
1..59 shift down to 0..58, matching mmseg's own ``reduce_zero_label=True``
handling for this exact variant).
"""

PASCAL_CONTEXT_CLASSES = [
    "aeroplane", "bag", "bed", "bedclothes", "bench", "bicycle", "bird",
    "boat", "book", "bottle", "building", "bus", "cabinet", "car", "cat",
    "ceiling", "chair", "cloth", "computer", "cow", "cup", "curtain", "dog",
    "door", "fence", "floor", "flower", "food", "grass", "ground", "horse",
    "keyboard", "light", "motorbike", "mountain", "mouse", "person",
    "plate", "platform", "pottedplant", "road", "rock", "sheep", "shelves",
    "sidewalk", "sign", "sky", "snow", "sofa", "diningtable", "track", "train",
    "tree", "truck", "tvmonitor", "wall", "water", "window", "wood",
]

assert len(PASCAL_CONTEXT_CLASSES) == 59, f"Expected 59 classes, got {len(PASCAL_CONTEXT_CLASSES)}"
