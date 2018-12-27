import math


class ErrorMonitor:
    def __init__(self, name):
        self.name = name
        self.count = 0
        self.err_quantization_inc = 0
        self.err_overflow_inc = 0
        self.minv, self.maxv = None, None

    def __call__(self, before_v, after_v, golden_v, reason='any'):
        self.minv = min(self.minv, before_v) if self.minv is not None else before_v
        self.maxv = max(self.maxv, before_v) if self.maxv is not None else before_v
        self.count = self.count + 1
        if reason == 'quantization':
            self.err_quantization_inc = self.err_quantization_inc + \
                                        abs(after_v - before_v)
        elif reason == 'overflow':
            self.err_overflow_inc = self.err_overflow_inc + \
                                    abs(after_v - before_v)
            pass
        else:
            raise ValueError('not supported reason')

    def __str__(self):
        return 'error monitor {}: [min/max: {} / {}]  [quantization: {} / {}]  [overflow: {} / {}]'.format(
            self.name,
            self.minv, self.maxv,
            self.err_quantization_inc, self.count, self.err_overflow_inc, self.count
        )
