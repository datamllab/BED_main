import numpy as np
import torch


def torch_approx_sigmoid(arr):
    tensor = torch.FloatTensor(arr) / (2 ** 14)
    return 1 / (1 + torch.pow(2, -1 * tensor))


def q_div(a, b):
    # a, b = int32, int32
    Q = 14
    temp = np.int64(a) << Q
    if (temp >= 0 and b >= 0) or (temp < 0 and b < 0):
        temp += (b >> 1)
    else:
        temp -= (b >> 1)
    return np.int32(temp // b)


# // saturate to range of int32_t
# int16_t sat16(int32_t x)
# {
# if (x > 0x7FFF) return 0x7FFF;
# else if (x < -0x8000) return -0x8000;
# else return (int16_t)x;
# }
#
def sat32(x):
    # x = int64_t
    x = np.int64(x)
    if x > 0x7FFFFFFF:
        return 0x7FFFFFFF
    elif x < -0x80000000:
        return -0x80000000
    else:
        return np.int32(x)


# int16_t q_mul(int16_t a, int16_t b)
# {
# int16_t result;
# int32_t temp;
#
# temp = (int32_t)a * (int32_t)b; // result type is operand's type
# // Rounding; mid values are rounded up
# temp += K;
# // Correct by dividing by base and saturate result
# result = sat16(temp >> Q);
#
# return result;
# }
def q_mul(a, b):
    # a, b = int32, int32
    Q = 14
    temp = np.int64(a) * np.int64(b)
    K = (1 << (Q - 1))
    temp += K
    result = sat32(temp >> Q)
    return result


# y_i = 2^(x_i/16384) / (2^(x_i/16384) + 1)
def approx_sigmoid(arr):
    out = []
    Q = 14
    for elmnt in arr:
        if elmnt < 0:
            elmnt = ~(elmnt - 1)
        else:
            pos_elmnt = elmnt
        shift = np.uint32(pos_elmnt) >> Q
        base = np.int32(1) << shift
        if elmnt > 0:
            tmp = q_div(base, base + 1)
        else:
            tmp = q_div(np.int32(1), base + 1)
        out.append(tmp)
    return out


# y_i = t / (1 + |t|)
# def approx_sigmoid_v2(arr):
#     out = []
#     Q = 14
#     for elmnt in arr:
#         if elmnt < 0:
#             abs_elmnt = ~(elmnt - 1)
#         else:
#             abs_elmnt = elmnt
#         numerator = np.uint32(elmnt) >> Q
#         denominator = 1 + (abs_elmnt >> Q)
#         tmp = q_div(numerator, denominator)
#         out.append(tmp)
#     return out


def sigmoid(arr):
    arr = np.asarray(arr)
    return 1 / (1 + np.exp(-1 * arr))


def q17p14(arr):
    return np.floor(arr * (2**14) + 0.5).astype(np.int32)


def sigmoid_lut(arr, y_q_num, low, high, resolution):
    out = []
    upper = q17p14((high - resolution))
    lower = q17p14(low)
    for e in arr:
        if e >= upper:
            res = y_q_num[255]
        elif e <= lower:
            res = y_q_num[0]
        else:
            idx = np.int32((e >> 10) + 128)
            y1 = y_q_num[idx]
            y2 = y_q_num[idx + 1]
            # res = y1 + ((y2 - y1) >> 10) * (e & 1023)
            slope = q_div(y2-y1, 1024)
            diff = np.int32(e & 1023)
            res = y1 + q_mul(slope, diff)
        out.append(res)
    return out


def generate_q_sigmoid(max_val=8, samples=256):
    size = max_val
    low, high = -size, size
    resolution = size / samples * 2
    pos_128 = np.arange(0, high, resolution, dtype=np.float32)
    neg_128 = np.arange(low, 0, resolution, dtype=np.float32)
    x = np.concatenate((neg_128, pos_128))
    real_sigmoid = sigmoid(x)
    # a1 = np.asarray((x, real_sigmoid)).transpose()
    # np.savetxt("sigmoid.csv", a1, delimiter=",")
    y_q_num = q17p14(real_sigmoid)
    # x_q_num = q17p14(x)
    # a2 = np.asarray((x_q_num, y_q_num)).transpose()
    # np.savetxt("q_sigmoid.csv", a2, delimiter=",")
    return y_q_num, low, high, resolution


def hex_format(q_sigmoid):
    hex_q_sigmoid = ['0x%08x,'%i for i in q_sigmoid]
    for i in [hex_q_sigmoid[c:c+8] for c in range(0, len(hex_q_sigmoid), 8) if c % 8 == 0]:
        print(' ', *i, '\\')

if __name__ == '__main__':
    q_sigmoid, l, h, resolution = generate_q_sigmoid()
    # hex_format(q_sigmoid)

    test_cases = (
        0,
        -8,
        7.9375,
        0.625,
        -0.625,
        0.0313,
        -0.0313,
        9,
        -9,
    )
    arr = [q17p14(t) for t in test_cases]
    val = sigmoid_lut(arr, q_sigmoid, l, h, resolution)
    print('test case:', test_cases)
    print('lut_sigmoid:', val)
    print('sigmoid:', q17p14(sigmoid(test_cases)))

    path = 'feature_map.npy'
    x = np.load(path)
    y_lut = sigmoid_lut(x, q_sigmoid, l, h, resolution)
    print('sigmoid_lut:', y_lut)
    y_sigmoid = q17p14(sigmoid(x / (2 ** 14))).tolist()
    print('sigmoid:', y_sigmoid)
