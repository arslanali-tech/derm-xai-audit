"""dermxai: is a skin-lesion classifier looking at the lesion? A mask-based audit of explanations."""

__version__ = "0.1.0"

# HAM10000 diagnosis codes in a fixed order; the index in this tuple is the class label.
CLASSES: tuple[str, ...] = ("akiec", "bcc", "bkl", "df", "mel", "nv", "vasc")
