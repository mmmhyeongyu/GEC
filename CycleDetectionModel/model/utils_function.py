import time
import pandas as pd
import numpy as np
from collections import defaultdict
from scipy import optimize
from scipy import ndimage
from tslearn.metrics import dtw
import matplotlib.pyplot as plt
import config

def inactive_phase(smooth, times):
    """
        define inactive phase in the data based on the slope and its duration

        :param smooth: np array of floats containing smoothed/cleaned data
        :param times: np array of timestamp of time interval of interest
        :return: array of array of int representing the index
    """
    smooth_d1 = [abs(smooth[i + 1] - smooth[i]) for i in range(len(smooth) - 1)] + [0]
    smooth_d2 = [smooth[i + 1] - smooth[i] for i in range(len(smooth) - 1)] + [0]

    # let's define the slope threshold to detect potential inactive phase
    up_threshold, down_threshold, t_diff = max(0.005, np.percentile(smooth_d1, 60)), np.percentile(smooth_d1, 10), 2
    cycles, index = [], 0
    while index < len(times):
        # absolute of slope within the threshold indicate inactive phase
        if (0 <= smooth_d1[index] <= up_threshold):
            stack = (index, smooth_d1[index], smooth[index], times[index])
            while True:
                if index + 1 < len(times) and smooth_d1[index + 1] <= up_threshold and abs(sum(smooth_d2[stack[0]:index + 1])) <= up_threshold:
                    index += 1
                else:
                    break
            # inactive if it lasts more than t_diff * X
            if (times[index] - times[stack[0]]) / np.timedelta64(1, 's') > t_diff * 600:
                value = np.mean(smooth[stack[0]:index + 1])
                if value <= max(0.3, np.percentile(smooth, 10)) * 1.1:
                    if cycles and (times[stack[0]] - times[cycles[-1][1]]) / np.timedelta64(1, 's') < t_diff * 60:
                        cycles[-1][1] = index
                    else:
                        cycles += [[stack[0], index]]
        index += 1
    return cycles


def inactive_phase_cleaning_normalization(smooth_, times_, upper_limit, window, so_range):
    """
        remove inactive phase from data of interest and normalize all values between 0 and 1.1 (0.1 is added to avoid division by 0)

        :param smooth_: np array of floats containing smoothed/cleaned data
        :param times_: np array of timestamp of time interval of interest
        :param upper_limit: int representing the percentile of selection to define the starting point of active phase
        :param window: int representing the time interval in minutes for rolling minimum approach
        :return: times: np array of timestamp of active time interval
        :return: ss: np array of floats containing active smoothed/cleaned data
        :return: inactive_cycles_: array of array of timestamp of inactive phase
    """

    if upper_limit > 60:  # for small/medium machine, we need to find inactive phase
        inactive_cycles = inactive_phase(smooth_, times_)

        inactive_value = []
        for i in inactive_cycles:
            inactive_value += [np.mean(smooth_[i[0]:i[1] + 1])]

        if config.plot:
            plt.figure(figsize=(25, 2))
            plt.plot(times_, smooth_, alpha=0.5, label='smooth')
            for i in inactive_cycles:
                plt.plot(times_[i[0]:i[1] + 1], smooth_[i[0]:i[1] + 1], linewidth=3)
            plt.title('inactive')
            plt.legend()
            plt.show()

        inactive_cycles_ = [[times_[i], times_[j]] for i, j in inactive_cycles]
        check = True
    else:  # for large machine, we don't need to detect inactive phase
        inactive_value, inactive_cycles_ = [], []

    if inactive_value != [] and check and np.median(inactive_value) < max(0.3, min(smooth_) + 0.05):
        inactive_value = np.mean(inactive_value)
        times, smooth = [], []
        inactive_cycles = [[0, 0]] + inactive_cycles + [[len(smooth_) - 1, len(smooth_)]]
        for index in range(1, len(inactive_cycles)):
            t = smooth_[inactive_cycles[index - 1][1] + 1:inactive_cycles[index][0]]
            #print(np.mean(t), inactive_value * 1.1, inactive_cycles[index])
            if t and np.mean(t) > inactive_value:
                times += times_[max(0, inactive_cycles[index - 1][1] - 1):inactive_cycles[index][0] + 2]
                smooth += smooth_[max(0, inactive_cycles[index - 1][1] - 1):inactive_cycles[index][0] + 2]
    else:
        times, smooth = times_, smooth_

    # rolling window normalization
    t_diff = 2  # seconds
    steps = window * 60 // t_diff
    start = 0
    times_t, smooth_t = times.copy(), smooth.copy()
    tt, ss, max_value = [], [], np.percentile(smooth_t, 98)
    while True:
        t_, s_ = times_t[start:start + steps], smooth_t[start:start + steps]
        min_value = np.percentile(smooth_t[max(0, start - steps // 2):start + steps], 10)
        ss += [(min(max(xx, min_value), max_value) - min_value) / (max_value - min_value + 0.0001) + 0.1 for xx in s_]
        start += steps
        if start >= len(times_t):
            break

    if config.plot and so_range:
        plt.figure(figsize=(25, 2))
        plt.plot(times, ss, label='original')
        plt.axvspan(so_range[0], so_range[1], color='k', alpha=0.2, label='so_range')
        plt.title("smoothed")
        plt.legend(loc='upper left')
        plt.show()
        
    return times, ss, inactive_cycles_

def cleaning(temp, smooth_factor, min_threshold):
    """
        remove inactive phase from data of interest and normalize all values between 0 and 1.1 (0.1 is added to avoid division by 0)

        :param smooth_: np array of floats containing smoothed/cleaned data
        :param times_: np array of timestamp of time interval of interest
        :param upper_limit: int representing the percentile of selection to define the starting point of active phase
        :param window: int representing the time interval in minutes for rolling minimum approach
        :return: times: np array of timestamp of active time interval
        :return: ss: np array of floats containing active smoothed/cleaned data
        :return: inactive_cycles_: array of array of timestamp of inactive phase
    """
    times, y = temp['timestamp'].values, temp['value'].values
    
    # 1. remove duplicates (within 0.5 seconds)
    # 2. let's fill in the blank (if the diff between consecutive points is greater than 2 seconds)
    # 3. normalize
    times_, y_ = [], []
    previous = [times[0], y[0]]
    for i in range(1, len(times)):
        while (times[i] - previous[0]) / np.timedelta64(1, 's') > 1:
            times_ += [previous[0]]
            y_ += [previous[1]]
            previous = [previous[0] + np.timedelta64(2,'s'), previous[1]]
        previous = [times[i], y[i]]
    if (times[i] - times_[-1]) / np.timedelta64(1, 's') > 0.5: # add last element if feasible
        times_ += [times[i]]
        y_ += [y[i]]


    y_ = ndimage.median_filter(np.array(y_), smooth_factor)
    # min_threshold cutting
    y_ = [max(min_threshold, yy) for yy in y_]
    m = np.percentile(y_, 99)
    # normalize
    y = [min(1, yy / m) for yy in y_]
    times = times_.copy()
    return times, y

def smoothing(smooth, step):
    if step > 0:
        smooth = [np.mean(smooth[max(i-step, 0):min(i+step, len(smooth))]) for i in range(len(smooth))]
    return smooth

def merger(cycles, max_cycle, smooth, times, min_value):
    # intention is to merge subset cycles to its superset
    output = []
    for c in cycles[::-1]:
        #if output:
        #    print(c, np.max(smooth[c[0]:c[1]]), min_value, output[-1][0], c[1])
        if np.max(smooth[c[0]:c[1]]) > min_value:
            if (times[c[1]] - times[c[0]]) / np.timedelta64(1, 's') < max_cycle:
                if output == [] or (output and output[-1][0] >= c[1]):
                    output += [c]
            else:
                print("too big", c, times[c[1]], times[c[0]], (times[c[1]] - times[c[0]]) / np.timedelta64(1, 's'))
    return output[::-1]

def active_phase(smooth, times, threshold, upper_limit, max_cycle, gap, min_value):
    smooth_d1 = [smooth[i + 1] - smooth[i] for i in range(len(smooth) - 1)] + [0]
    up_threshold, down_threshold, t_diff = np.percentile(smooth_d1, upper_limit), np.percentile(smooth_d1,
                                                                                                100 - upper_limit + 10), 2
    #print(up_threshold, down_threshold)

    stack, cycles, index, check = [], [], 0, False
    while index < len(times):
        if (smooth_d1[index] > up_threshold):
            stack += [(max(index - 1, 0), smooth_d1[index], smooth[index - 1], times[index - 1])]
            # if it is going upward, let's continue searching
            while True:
                if index + 1 < len(times) and smooth_d1[index + 1] >= 0:
                    index += 1
                else:
                    break
            check = False
        elif smooth_d1[index] < down_threshold or (stack and smooth[index] <= stack[-1][2]):
            # if it is going downward, let's continue searching while slope is negative
            temp_downward = index + 0  # make sure it doesn't overwrite
            lookback = 0
            while True:
                threshold_ = 1 / ((smooth[index] * 10) ** 1.5) * (threshold - 1) + 1
                if index + 1 < len(times) and smooth[index] / smooth[index + 1] > threshold_:
                    index += 1
                    lookback = 1
                else:
                    break
            pool = smooth[index: index + gap // t_diff + 1]
            if smooth[index] / min(pool) <= 1.1:  # make sure we are looking at the minimum including the neighbors
                # check if downward was truly downward
                if smooth[temp_downward] / smooth[index] < threshold:
                    index = temp_downward + 1
                wrap = []
                while stack:
                    a, b, c, d = stack.pop(-1)
                    if index >= len(smooth):
                        index = len(smooth) - 1
                    threshold_ = 1 / ((c * 10) ** 1.5) * (threshold - 1) + 1
                    if smooth[min(index, len(smooth) - 1)] / c <= threshold_ and max_cycle > (
                            times[index] - d) / np.timedelta64(1, 's') >= gap:
                        wrap = (a, b, c, d)
                    else:
                        stack += [(a, b, c, d)]
                        break
                if wrap != []:
                    aa, b, c, d = wrap
                    if cycles and (times[aa] - times[cycles[-1][1]]) / np.timedelta64(1, 's') < gap:  # and smooth[a] > smooth[cycles[-1][0]]:
                        cycles[-1] = [min(aa, cycles[-1][0]), max(index, cycles[-1][1])]
                    else:
                        cycles += [[aa, index]]
                    check = smooth[index] > smooth[a]
        index += 1
    return merger(cycles, max_cycle, smooth, times, min_value)

def cycle_generate(cycles_, times, smooth, inactive_cycles_, so_range):
    cycles = []
    for i, j in cycles_:
        temp = []
        for k in range(i, j + 1):
            if not any(aa < times[k] < bb for aa, bb in inactive_cycles_):
                temp += [[times[k], smooth[k], k]]
            else:  # if any of time is within inactive cycles, let's break!
                if temp:
                    cycles += [temp]
                temp = []
        if temp:
            cycles += [temp]

    cycles = [x for x in cycles if min([y[1] for y in x]) <= 0.3]
    if so_range:
        cycles = [cy for cy in cycles if (cy[0][0]>=so_range[0]) or (cy[0][0]<so_range[0] and cy[-1][0] >= so_range[0])]

    if config.plot and so_range:
        plt.figure(figsize=(25, 2))
        plt.plot(times, smooth, alpha=0.5, label='smooth')
        for j, i in enumerate(cycles):
            xx, yy = [aa[0] for aa in i], [aa[1] for aa in i]
            #print(j, xx[0], xx[-1])
            plt.plot(xx, yy, linewidth=3)
        plt.axvspan(so_range[0], so_range[1], color='k', alpha=0.2, label='so_range')
        plt.legend(loc='upper left')
        plt.title("Active Phase Candidates")
        plt.show()
    return cycles

