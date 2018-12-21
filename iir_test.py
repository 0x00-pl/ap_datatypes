import math

from ac_datatype import AcFixed
import monitor

iir3_m_r1 = monitor.ErrorMonitor('irr3_m_r1')
iir3_m_t1 = monitor.ErrorMonitor('irr3_m_t1')
iir3_m_r2 = monitor.ErrorMonitor('irr3_m_r2')
iir3_m_t2 = monitor.ErrorMonitor('irr3_m_t2')
iir3_m_r3 = monitor.ErrorMonitor('irr3_m_r3')
iir3_m_t3 = monitor.ErrorMonitor('irr3_m_t3')
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


def apply_iir3_m(seq, b3, a3, mr=None, mt=None):
    regi = 2
    argi = 2
    ret = []
    b3 = [AcFixed(16, argi, True, i) for i in b3]
    a3 = [AcFixed(16, argi, True, i) for i in a3]
    print('quantlized a3 b3:', a3, b3)
    t = AcFixed(16, 1, True, 0)
    reg1 = AcFixed(16, regi, True, 0)
    reg2 = AcFixed(16, regi, True, 0)
    for i in seq:
        t = b3[0] * i + reg1
        t = t.to_fixed(16, 1, True, AcFixed.QuantizationMode.RND, AcFixed.OverflowMode.SAT, mt)
        reg1 = b3[1] * i + reg2 - a3[1] * t
        reg1 = reg1.to_fixed(16, regi, True, AcFixed.QuantizationMode.RND, AcFixed.OverflowMode.SAT, mr)
        reg2 = b3[2] * i - a3[2] * t
        reg2 = reg2.to_fixed(16, regi, True, AcFixed.QuantizationMode.RND, AcFixed.OverflowMode.SAT, mr)

        r = t
        ret.append(r)

    return ret


def test_iir3x3(seq):
    b3 = [0.9819946289062, -1.963989257812, 0.9819946289062]
    a3 = [1, -1.963989257812, 0.964599609375]

    r1 = apply_iir3_m(seq, b3, a3, iir3_m_r1, iir3_m_t1)
    # r1 = apply_iir3_m(r1, b3, a3, iir3_m_r2, iir3_m_t2)
    # r1 = apply_iir3_m(r1, b3, a3, iir3_m_r3, iir3_m_t3)
    # r1 = apply_iir3_m(r1, b3, a3, iir3_m_r3, iir3_m_t3)
    # r1 = apply_iir3_m(r1, b3, a3, iir3_m_r3, iir3_m_t3)

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



if __name__ == '__main__':
    seq = [i * 2 * math.pi / 48000 * 6000 for i in range(4800)]
    seq = [math.sin(i) for i in seq]
    seq = [1] * len(seq)
    seq = [AcFixed(16, 1, True, i) for i in seq]

    print('seq err db', err_sqr_db(seq))
    r1 = test_iir3x3(seq)
    # r2 = test_iir7(seq)
    print()
    print('seq eng', eng(seq))
    print('ret eng', eng(r1))
    # print('eng', eng(r2))
    print()
    print('seq values:', [i for i in seq[1100:1200]])
    print('ret values:', [i for i in r1[1100:1200]])
    # print([i for i in r2[1100:1200]])
    print()
    print(iir3_m_r1)
    print(iir3_m_t1)
    print(iir3_m_r2)
    print(iir3_m_t2)
    print(iir3_m_r3)
    print(iir3_m_t3)
    # print(iir7_m_r)
    # print(iir7_m_t)
