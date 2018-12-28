import math

import audio_export
from ac_datatype import AcFixed
import monitor

iir3_m_pa = monitor.ErrorMonitor('iir3_m_pa')
iir3_m_pb = monitor.ErrorMonitor('iir3_m_pb')

iir3_m_r1 = monitor.ErrorMonitor('irr3_m_r1')
iir3_m_t1 = monitor.ErrorMonitor('irr3_m_t1')
iir3_m_r2 = monitor.ErrorMonitor('irr3_m_r2')
iir3_m_t2 = monitor.ErrorMonitor('irr3_m_t2')
iir3_m_r3 = monitor.ErrorMonitor('irr3_m_r3')
iir3_m_t3 = monitor.ErrorMonitor('irr3_m_t3')
iir3_m_r4 = monitor.ErrorMonitor('irr3_m_r4')
iir3_m_t4 = monitor.ErrorMonitor('irr3_m_t4')
iir3_m_r5 = monitor.ErrorMonitor('irr3_m_r5')
iir3_m_t5 = monitor.ErrorMonitor('irr3_m_t5')
iir7_m_r = monitor.ErrorMonitor('irr7_m_r')
iir7_m_t = monitor.ErrorMonitor('irr7_m_t')


# import numpy as np

def apply_iir3(seq, b3, a3, mr=None, mt=None):
    ret = []
    b3 = [AcFixed(16, 2, True, i) for i in b3]
    a3 = [AcFixed(16, 2, True, i) for i in a3]
    print('quantlized a3 b3:', a3, b3)
    t = AcFixed(16, 1, True, 0)
    reg1 = AcFixed(16, 1, True, 0)
    reg2 = AcFixed(16, 1, True, 0)
    for i in seq:
        t = a3[0] * i - a3[1] * reg1 - a3[2] * reg2
        r = b3[0] * t + b3[1] * reg1 + b3[2] * reg2
        # print(t.value, end=' ')
        r = r.to_fixed(16, 1, True, AcFixed.QuantizationMode.RND, AcFixed.OverflowMode.SAT, mr)
        ret.append(r)
        reg2 = reg1
        reg1 = t.to_fixed(16, 13, True, AcFixed.QuantizationMode.RND, AcFixed.OverflowMode.SAT, mt)

    return ret


def apply_iir3_m(seq, b3, a3, arg_pow, mr=None, mt=None):
    ret = []
    b3 = [i / (2 ** arg_pow) for i in b3]
    a3 = [i / (2 ** arg_pow) for i in a3]
    b3 = [AcFixed(32, 1, True, i) for i in b3]
    a3 = [AcFixed(32, 1, True, i) for i in a3]
    # print('quantlized a3 b3:', a3, b3)
    t = AcFixed(16, 1, True, 0)
    reg1 = AcFixed(16, 1, True, 0)
    reg2 = AcFixed(16, 1, True, 0)
    for i in seq:
        t = b3[0] * i + reg1
        t = t << arg_pow
        r = t.to_fixed(16, 1, True, AcFixed.QuantizationMode.RND, AcFixed.OverflowMode.SAT, mt)
        reg1 = b3[1] * i + reg2 - a3[1] * r
        reg1 = reg1.to_fixed(32, 1, True, AcFixed.QuantizationMode.RND, AcFixed.OverflowMode.SAT, mr)
        reg2 = b3[2] * i - a3[2] * r
        reg2 = reg2.to_fixed(32, 1, True, AcFixed.QuantizationMode.RND, AcFixed.OverflowMode.SAT, mr)

        ret.append(r)

    return ret


def test_iir3x3(seq):
    # b3 = [0.982, -1.964, 0.982]
    # b3 = [i / 1.3 for i in b3]
    # a3 = [1, -1.964, 0.9646]
    # sos = [
    #     [1, - 1.99993896484375, 1, 1, - 1.99774169921875, 0.997863769531250],
    #     [1, - 1.99993896484375, 1, 1, - 1.99145507812500, 0.991638183593750],
    #     [1, - 2, 1, 1,                - 1.97668457031250, 0.976989746093750],
    #     [1, - 1, 0, 1,                - 0.979125976562500, 0]
    # ]
    # g = [
    #     0.620239257812500,
    #     0.995788574218750,
    #     0.988403320312500,
    #     1.59375
    # ]
    sos = [
        [1, - 1.99993896484375, 1, 1, - 1.99768066406250, 0.997802734375000],
        [1, - 2, 1, 1, - 1.99127197265625, 0.991455078125000],
        [1, - 2, 1, 1, - 1.97631835937500, 0.976623535156250],
        [1, - 1, 0, 1, - 0.978881835937500, 0]
    ]
    g = [
        0.612609863281250,
        0.995666503906250,
        0.988220214843750,
        1.61322021484375
    ]
    a = []
    b = []
    for ab, ig in zip(sos, g):
        b3 = ab[:3]
        b3 = [i * ig for i in b3]
        a3 = ab[3:]
        b.append(b3)
        a.append(a3)

    arg_pow = math.ceil(math.log2(max(1, *[abs(i) for i in a3], *[abs(i) for i in b3])))

    # print('min eng', eng(seq))
    r1 = apply_iir3_m(seq, b[0], a[0], 1, iir3_m_r1, iir3_m_t1)
    # print('min eng', eng(r1))
    r1 = apply_iir3_m(r1, b[1], a[1], 1, iir3_m_r2, iir3_m_t2)
    # print('min eng', eng(r1))
    r1 = apply_iir3_m(r1, b[2], a[2], 1, iir3_m_r3, iir3_m_t3)
    # print('min eng', eng(r1))
    r1 = apply_iir3_m(r1, b[3], a[3], 1, iir3_m_r4, iir3_m_t4)
    # print('min eng', eng(r1))
    # r1 = apply_iir3_m(r1, [2, 0, 0], [1, 0, 0], 2, iir3_m_r5, iir3_m_t5)
    # r1 = apply_iir3_m(r1, b3, a3, arg_pow, iir3_m_r3, iir3_m_t3)
    # r1 = apply_iir3_m(r1, [1.3, 0, 0], [1, 0, 0], arg_pow, iir3_m_r3, iir3_m_t3)
    # r1 = apply_iir3_m(r1, b3, a3, arg_pow, iir3_m_r3, iir3_m_t3)
    # r1 = apply_iir3_m(r1, [1.3, 0, 0], [1, 0, 0], arg_pow, iir3_m_r2, iir3_m_t2)
    # r1 = apply_iir3_m(r1, b3, a3, arg_pow, iir3_m_r3, iir3_m_t3)
    # r1 = apply_iir3_m(r1, [1.3, 0, 0], [1, 0, 0], arg_pow, iir3_m_r2, iir3_m_t2)
    # r1 = apply_iir3_m(r1, b3, a3, arg_pow, iir3_m_r3, iir3_m_t3)

    print('ret err db', err_sqr_db(r1))
    return r1


# def apply_iir7(seq, b7, a7, I=1):
#     ret = []
#     b7 = [AcFixed(16, I, True, i) for i in b7]
#     a7 = [AcFixed(16, I, True, i) for i in a7]
#     t = AcFixed(16, I, True, 0)
#     reg1 = AcFixed(16, I, True, 0)
#     reg2 = AcFixed(16, I, True, 0)
#     reg3 = AcFixed(16, I, True, 0)
#     reg4 = AcFixed(16, I, True, 0)
#     reg5 = AcFixed(16, I, True, 0)
#     reg6 = AcFixed(16, I, True, 0)
#     for i in seq:
#         t = a7[0] * i - a7[1] * reg1 - a7[2] * reg2 - a7[3] * reg3 - a7[4] * reg4 - a7[5] * reg5 - a7[6] * reg6
#         r = b7[0] * t + b7[1] * reg1 + b7[2] * reg2 + b7[3] * reg3 + b7[4] * reg4 + b7[5] * reg5 + b7[6] * reg6
#         r = r.to_fixed(16, 1, True, AcFixed.QuantizationMode.RND, AcFixed.OverflowMode.SAT, iir7_m_r)
#         ret.append(r)
#         reg6 = reg5
#         reg5 = reg4
#         reg4 = reg3
#         reg3 = reg2
#         reg2 = reg1
#         reg1 = t.to_fixed(16, I, True, AcFixed.QuantizationMode.RND, AcFixed.OverflowMode.SAT, iir7_m_t)
#
#     return ret


def err_sqr_db(seq):
    s = 0
    for i in seq:
        err = abs(i.value - i.dequant())
        s = s + err * err

    s = math.sqrt(s)
    db = 20 * math.log10(s + 1e-6)
    return db


def eng(seq):
    return sum([(i * i).to_fixed(64, 32, True) for i in seq])


# def test_iir7(seq):
#     b7 = [0.9470, -5.6817, 14.2043, -18.9390, 14.2043, -5.6817, 0.9470]
#     a7 = [1.0000, -5.8920, 14.4656, -18.9424, 13.9535, -5.4822, 0.8975]
#
#     r1 = apply_iir7(seq, b7, a7, 7)
#     print('err 7', err_sqr_db(r1))
#     return r1
#

def test_all_freq():
    for freq in range(10, 200, 10):
        seq = [math.sin(i * 2 * math.pi / 48000 * freq) for i in range(1000)]
        seq = [AcFixed(16, 1, True, i) for i in seq]
        print('seq {} err db'.format(freq), err_sqr_db(seq))
        r1 = test_iir3x3(seq)
        print('ret {} eng'.format(freq), eng(r1), eng(seq))

    for freq in range(200, 24000, 100):
        seq = [math.sin(i * 2 * math.pi / 48000 * freq) for i in range(1000)]
        seq = [AcFixed(16, 1, True, i) for i in seq]
        print('seq {} err db'.format(freq), err_sqr_db(seq))
        r1 = test_iir3x3(seq)
        print('ret {} eng'.format(freq), eng(r1), eng(seq))


def plot_wave(w, post_fix=''):
    try:
        import matlab
        from matlab import engine
        enge = engine.connect_matlab()
        enge.workspace['quanted' + post_fix] = matlab.double([i.dequant() for i in w])
        enge.workspace['value' + post_fix] = matlab.double([i.value for i in w])
        enge.eval('signalAnalyzer', nargout=0)
    except:
        pass


if __name__ == '__main__':
    # test_all_freq()
    # exit()
    # seq = [i * 2 * math.pi / 48000 * 150 for i in range(4800)]
    # seq = [math.sin(i)/4 for i in seq]
    # seq = [1] * len(seq)
    # seq = [1e-4*(random.random()-0.5) for i in seq]
    seq = [0] * 1000 + audio_export.data
    seq = [AcFixed(16, 1, True, v / 2 + math.sin(i * 2 * math.pi * 50 / 48000) / 4) for v, i in
           zip(seq, range(len(seq)))]

    print('seq err db', err_sqr_db(seq))
    r1 = test_iir3x3(seq)
    # r2 = test_iir7(seq)
    plot_wave(r1, '32m')
    print()
    print('seq eng', eng(seq))
    print('ret eng', eng(r1))
    # print('eng', eng(r2))
    print()
    print('seq values:', [i for i in seq[1100:1200]])
    print('ret values:', [i for i in r1[1100:1200]])
    # print([i for i in r2[1100:1200]])
    print()
    print(iir3_m_t1)
    print(iir3_m_r1)
    print(iir3_m_t2)
    print(iir3_m_r2)
    print(iir3_m_t3)
    print(iir3_m_r3)
    print(iir3_m_t4)
    print(iir3_m_r4)
    print(iir3_m_t5)
    print(iir3_m_r5)
    # print(iir7_m_r)
    # print(iir7_m_t)
