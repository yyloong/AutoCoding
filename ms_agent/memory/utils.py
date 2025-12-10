# Copyright (c) Alibaba, Inc. and its affiliates.
from .default_memory import DefaultMemory
from .diversity import Diversity
from .mem0ai import Mem0Memory
from .statememory import ExactStateMemory

memory_mapping = {
    'default_memory': DefaultMemory,
    'statememory': ExactStateMemory,
    'mem0': Mem0Memory,
    'diversity': Diversity,
}
