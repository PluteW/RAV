from abc import ABC, abstractmethod

from PIL import Image


class Model(ABC):
    def __init__(self):
        super().__init__()
        self.name = "UNNAMED"

    @abstractmethod
    def answer(self, vision:Image, touch:Image):
        pass