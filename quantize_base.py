

class Quantize:
    def __init__(self, value):
        self.value = value

    @staticmethod
    def low_bits(value, width):
        return int(value) and ((1<<width)-1)