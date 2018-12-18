from enum import Enum

from quantize_base import Quantize


class AcInt(Quantize):
    def __init__(self, width, signed, value=None, qvalue=None):
        self.W = width
        self.S = signed
        if value is not None:
            Quantize.__init__(self, value)
            self.quant(qvalue if qvalue is not None else value)

    def quant(self, value):
        if self.S:
            self.mem = int(value)
            if value < (-(1 << (self.W - 1))) or value > ((1 << (self.W - 1)) - 1):
                raise ValueError('out of range, {} W={} S={}'.format(value, self.W, self.S))
        else:
            self.mem = int(value)
            if value < 0 or value > ((1 << self.W) - 1):
                raise ValueError('out of range, {} W={} S={}'.format(value, self.W, self.S))

    def dequant(self):
        return self.mem

    def __str__(self):
        return str(self.dequant()) + '({})'.format(self.value - self.dequant())

    def __add__(self, other):
        sw = self.W + (1 if not self.S and other.S else 0)
        ow = other.W + (1 if not other.S and self.S else 0)
        s = self.S or other.S
        ret = AcInt(max(sw, ow) + 1, s, self.value + other.value, self.dequant() + other.dequant())
        return ret

    def __sub__(self, other):
        sw = self.W + (1 if not self.S and other.S else 0)
        ow = other.W + (1 if not other.S and self.S else 0)
        s = True
        ret = AcInt(max(sw, ow) + 1, s, self.value - other.value, self.dequant() - other.dequant())
        return ret

    def __mul__(self, other):
        s = self.S or other.S
        ret = AcInt(self.W + other.W, s, self.value * other.value, self.dequant() * other.dequant())
        return ret

    def __truediv__(self, other):
        return self // other

    def __floordiv__(self, other):
        s = self.S or other.S
        ret = AcInt(self.W + other.W, s, self.value / other.value, self.dequant() / other.dequant())
        return ret

    def __and__(self, other):
        sw = self.W + (1 if not self.S and other.S else 0)
        ow = other.W + (1 if not other.S and self.S else 0)
        s = self.S or other.S
        ret = AcInt(max(sw, ow), s, int(self.value) & int(other.value), self.dequant() & other.dequant())
        return ret

    def __or__(self, other):
        sw = self.W + (1 if not self.S and other.S else 0)
        ow = other.W + (1 if not other.S and self.S else 0)
        s = self.S or other.S
        ret = AcInt(max(sw, ow), s, int(self.value) | int(other.value), self.dequant() | other.dequant())
        return ret

    def __xor__(self, other):
        sw = self.W + (1 if not self.S and other.S else 0)
        ow = other.W + (1 if not other.S and self.S else 0)
        s = self.S or other.S
        ret = AcInt(max(sw, ow), s, int(self.value) ^ int(other.value), self.dequant() ^ other.dequant())
        return ret


class AcFixed(Quantize):
    class QuantizationMode(Enum):
        TRN, TRN_ZERO, RND, RND_ZERO, RND_INF, RND_MIN_INF, RND_CONV, RND_CONV_ODD = range(8)

    class OverflowMode(Enum):
        WARP, SAT, SAT_ZERO, SAT_SYM = range(4)

    def __init__(self, width, int_width, signed, value=None, qvalue_shifted=None):
        self.W = width
        self.I = int_width
        self.S = signed
        if value is not None:
            Quantize.__init__(self, value)
            f = self.W - self.I
            self.quant(qvalue_shifted if qvalue_shifted is not None else value * (1 << f))

    def quant(self, qvalue_shifted):
        f = self.W - self.I
        if self.S:
            self.mem = int(qvalue_shifted)
            if qvalue_shifted < (-(1 << (self.W - 1))) or qvalue_shifted > ((1 << (self.W - 1)) - 1):
                raise ValueError('out of range, {} {} W={} I={} S={}'.format(qvalue_shifted/(1<<f), qvalue_shifted, self.W, self.I, self.S))
        else:
            self.mem = int(qvalue_shifted)
            if qvalue_shifted < 0 or qvalue_shifted > ((1 << self.W) - 1):
                raise ValueError('out of range, {} {} W={} I={} S={}'.format(qvalue_shifted/(1<<f), qvalue_shifted, self.W, self.I, self.S))

    def dequant(self):
        f = self.W - self.I
        return self.mem / (1 << f)

    def to_fixed(self, width, int_width, signed, quantization_mode, overflow_mode):
        sf = self.W-self.I
        of = width-int_width
        if of < sf:
            # needs quantization
            if quantization_mode == AcFixed.QuantizationMode.TRN:
                value_shifted = self.mem>>(sf-of)
            elif quantization_mode == AcFixed.QuantizationMode.RND:
                value_shifted = int(round(self.mem/(1<<(sf-of))))
            else:
                raise NotImplementedError()
        else:
            value_shifted = self.mem<<(of-sf)

        if int_width < self.I or signed != self.S:
            # needs overflow
            if overflow_mode == AcFixed.OverflowMode.WARP:
                value_shifted = Quantize.low_bits(value_shifted, width)
            elif overflow_mode == AcFixed.OverflowMode.SAT:
                minv = -(1 << (width - 1)) if signed else 0
                maxv = ((1 << (width - 1)) - 1) if signed else ((1 << width) - 1)
                value_shifted = max(min(value_shifted, maxv), minv)
            else:
                raise NotImplementedError()
        else:
            value_shifted = value_shifted

        return AcFixed(width, int_width, signed, self.value, value_shifted)


    def __str__(self):
        return str(self.dequant()) + '({})'.format(self.value - self.dequant())

    def __add__(self, other):
        s = self.S or other.S
        si = self.I + (1 if not self.S and other.S else 0)
        oi = other.I + (1 if not other.S and self.S else 0)
        i = max(si, oi) + 1
        sw = self.W - self.I
        ow = other.W - other.I
        w = i + max(sw, ow)
        sf = self.W - self.I
        of = other.W - other.I
        rf = w - i
        mem = (self.mem << (rf - sf)) + (other.mem << (rf - of))
        ret = AcFixed(w, i, s, self.value + other.value, mem)
        return ret

    def __sub__(self, other):
        s = True
        si = self.I + (1 if not self.S and other.S else 0)
        oi = other.I + (1 if not other.S and self.S else 0)
        i = max(si, oi) + 1
        sw = self.W - self.I
        ow = other.W - other.I
        w = i + max(sw, ow)
        sf = self.W - self.I
        of = other.W - other.I
        rf = w - i
        mem = (self.mem << (rf - sf)) - (other.mem << (rf - of))
        ret = AcFixed(w, i, s, self.value - other.value, mem)
        return ret

    def __mul__(self, other):
        s = self.S or other.S
        i = self.I + other.I
        w = self.W + other.W
        mem = self.mem * other.mem
        ret = AcFixed(w, i, s, self.value * other.value, mem)
        return ret

    def __truediv__(self, other, quantization_mode=QuantizationMode.TRN):
        s = self.S or other.S
        i = self.I + (other.W - other.I) + (1 if other.S else 0)
        w = self.W + max(other.W - other.I, 0) + (1 if other.S else 0)
        sf = self.W - self.I
        of = other.W - other.I
        rf = w - i
        val = (self.mem << (rf - sf)) / (other.mem << (rf - of))
        val_shifted = val * (1<<rf)
        if quantization_mode == AcFixed.QuantizationMode.TRN:
            mem = int(val_shifted)
        else:
            raise NotImplementedError()

        ret = AcFixed(w, i, s, self.value / other.value, mem)
        return ret


def test():
    a = AcInt(8, True, -127)
    b = AcInt(8, False, 255)
    print(a + b)
    print(a - b)
    print(a * b)
    print(a / b)
    print(a & b)
    print(a | b)
    print(a ^ b)


    c = AcFixed(8, 2, True, 0.125)
    d = c.to_fixed(6, 2, False, AcFixed.QuantizationMode.RND, AcFixed.OverflowMode.SAT)
    print(c,d)
    print(c + d)
    print(c - d)
    print(c * d)
    print(c / d)


if __name__ == '__main__':
    test()
