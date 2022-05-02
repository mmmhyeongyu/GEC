import pandas as pd
import numpy as np
from collections import defaultdict
from scipy import optimize
from scipy import ndimage
from tslearn.metrics import dtw
import json
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import config
import time


def func(x, *p):
    return np.poly1d(p)(x)


def poly_smooth(x, yy, cycle_parameter):
    # yy_ = [0]*parameter['n'] + yy + [0]*parameter['n']
    x_ = list(range(len(x)))  # + 2*parameter['n']))
    for i in range(cycle_parameter['n']):
        yy[i], yy[-(i + 1)] = 0, 0
    sigma = np.ones(len(x_))
    sigma[[0, -1]] = 0.01
    p1, _ = optimize.curve_fit(func, x_, yy, (0,) * cycle_parameter['order'], sigma=sigma, maxfev=8000)
    smoothed = func(np.linspace(0, len(x_), cycle_parameter['L']), *p1)
    return smoothed


def compute_euclidean(x1, x2):
    return np.linalg.norm(x1 - x2)


def compute_dtw(x1, x2):
    return dtw(x1, x2)


def time_diff_calculator(x, cycles):
    t1, t2 = cycles[x][0][0], cycles[x][-1][0]
    return (t2 - t1) / np.timedelta64(1, 's')


def pattern_history_parse(machine, order_number):
    machine = machine.replace('/', '')
    patterns, candidates, start_time, previous_id, first_phase, last_phase = {}, {}, "", [], {}, {}
    try:
        with open('./history/' + machine + '_history.json', 'r') as fin:
            past = json.load(fin)
        pattern_index = 1
        for i in past.keys():
            # if past[i]['order_number'] == order_number:
            if i != "last_active_timestamp":
                i = int(i)
                pattern_index = max(i, pattern_index)
                patterns[i] = past[str(i)]['info']
                patterns[i][2] = [tuple(xx) for xx in patterns[i][2]]
                candidates[i] = [np.array(past[str(i)]['pattern'][0])]
                first_phase[i] = [past[str(i)]['initial'][0]]
                last_phase[i] = [past[str(i)]['last'][0]]
                previous_id += [i]
        pattern_index += 1
        start_time = past['last_active_timestamp']
    except:
        pattern_index = 1
    return patterns, candidates, pattern_index, start_time, first_phase, last_phase, previous_id


def store_cycle_output_and_plot(output, patterns, candidates, machine, complete_cycle, times, ss, cycles, first_phase, last_phase, material, order_number, store=True):
    machine = machine.replace('/', '')
    if store:
        o = {}
        last_index = 0
        for i in output:
            #print(i)
            l = len(last_phase[i])
            o[i] = {}
            o[i]["order_number"] = [order_number]
            o[i]["info"] = [patterns[i][0], [np.mean(patterns[i][1])],
                            [(-1 - (patterns[i][2][-1][1] - patterns[i][2][-1][0]), -1)], [np.mean(patterns[i][3])],
                            [sum(patterns[i][4][-3:])]]
            o[i]["pattern"] = [[sum(candidates[i][ii][j] for ii in range(l)) / l for j in range(len(candidates[i][0]))]]
            o[i]["initial"] = [first_phase[i][-1]]
            o[i]["last"] = [last_phase[i][-1]]
            o[i]["material"] = [material]
            last_index = max(last_index, patterns[i][2][-1][1])
        #print(cycles[last_index][-1][0])
        o['last_active_timestamp'] = str(pd.to_datetime(cycles[last_index][-1][0]))
        # print(o)

        try:
            with open('./history/' + machine + '_history.json', 'r') as fin:
                t = json.load(fin)
            for i in output:
                l = len(last_phase[i])
                if i in t:
                    t[i]["order_number"] += [order_number]
                    if material not in t[i]["material"]:
                        t[i]["material"] += ["material"]
                else:
                    t[i] = {}
                    t[i]["order_number"] = [order_number]
                    t[i]["material"] = [material]
                t[i]["info"] = [patterns[i][0], [np.mean(patterns[i][1])],
                                [(-1 - (patterns[i][2][-1][1] - patterns[i][2][-1][0]), -1)], [np.mean(patterns[i][3])],
                                [sum(patterns[i][4][-3:])]]
                t[i]["pattern"] = [
                    [sum(candidates[i][ii][j] for ii in range(l)) / l for j in range(len(candidates[i][0]))]]
                t[i]["initial"] = [first_phase[i][-1]]
                t[i]["last"] = [last_phase[i][-1]]
                # t[i]["initial"] = [[sum(first_phase[i][ii][j] for ii in range(l))/l for j in range(len(first_phase[i][0]))]]
                # t[i]["last"] = [[sum(last_phase[i][ii][j] for ii in range(l))/l for j in range(len(last_phase[i][0]))]]
                last_index = max(last_index, patterns[i][2][-1][1])
            t['last_active_timestamp'] = str(pd.to_datetime(cycles[last_index][-1][0]))
        except:
            print("First time running")

    if store:
        # store output
        with open('./history/' + machine + '_history.json', 'w') as fout:
            try:
                json.dump(t, fout)
            except:
                json.dump(o, fout)

        # store output
        # with open('./history/' + machine + '_' + order_number + '_' + str(times[0])[:10] + '_history.json', 'w') as fout:
        #     json.dump(o, fout)

    # complete_cycle
    # if config.plot and not store:
    #     ww = complete_cycle
    #     # ww = final_cleaning(complete_cycle, cycles, output, patterns)
    #     plt.figure(figsize=(25, 2))
    #     plt.plot(times, ss, alpha=0.4, label='smooth')
    #     for i in sorted(ww):
    #         temp1 = sum(cycles[i[0]:i[1] + 1], [])  # concatenate
    #         xx, yy = [xx[0] for xx in temp1], [xx[1] for xx in temp1]
    #         plt.plot(xx, yy, linewidth=3)
    #     plt.legend()
    #     plt.show()
    #     # complete_cycle

    if config.plot and not store:
        ww = sorted([cy for cy in complete_cycle if cy[1] != -1])
        cycle_idx = 0
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=pd.to_datetime(times), y=ss, mode='lines', opacity=0.3, name='smooth', showlegend=False))
        for _, c in enumerate(ww):
            temp1 = sum(cycles[c[0]:c[1] + 1], [])  # concatenate
            xx, yy = pd.to_datetime([xx[0] for xx in temp1]), [xx[1] for xx in temp1]
            center_idx, max_yy = int(len(xx)//2), max(yy)
            fig.add_trace(go.Scatter(x=xx, y=yy, mode='lines', name=f'{cycle_idx}', line=dict(width=3)))
            fig.add_trace(go.Scatter(x=[xx[center_idx]], y=[max_yy*1.1], mode='markers+text', text=f'{cycle_idx}',
                            textposition='middle center', textfont=dict(family='bold', color='black'),
                            marker=dict(color='gray',opacity=0.2,size=15, line=dict(color='black', width=2)), showlegend=False))
            cycle_idx += 1
        fig.update_xaxes(showline=True, linewidth=1, linecolor='black', mirror=True, showgrid=False, rangeslider_visible=True)
        fig.update_yaxes(showline=True, linewidth=1, linecolor='black', mirror=True, showgrid=False)
        fig.update_layout(autosize=False, plot_bgcolor='white', width=1500, height=200, margin=dict(l=5,r=5,b=10,t=10,pad=2))
        fig.show()
        # time.sleep(2)

        # # ww = final_cleaning(complete_cycle, cycles, output, patterns)
        # plt.figure(figsize=(25, 2))
        # plt.plot(times, ss, alpha=0.4, label='smooth')
        # for i in sorted(ww):
        #     temp1 = sum(cycles[i[0]:i[1] + 1], [])  # concatenate
        #     xx, yy = [xx[0] for xx in temp1], [xx[1] for xx in temp1]
        #     plt.plot(xx, yy, linewidth=3)
        # plt.legend()
        # plt.show()


def overlapped(x, y):
    # since it's small, let's just use brute force
    x, y = sorted(x), sorted(y)
    x1, x2 = x[0][0], x[-1][1]
    y1, y2 = y[0][0], y[-1][1]
    if y2 < x1 or y1 > x2:
        return True
    return False


def gap_size(intervals):
    output = 0
    for i in range(1, len(intervals)):
        output += intervals[i][0] - intervals[i - 1][1] - 1
    return output


def final_cleaning(complete_cycle, cycles, output, patterns):
    ids = {}
    for i in output:
        for i_ in patterns[i][2]:
            # print(i_)
            ids[i_] = i
    ww = sorted([x for x in complete_cycle if x[1] != -1])
    # print(ww, ids)
    # the main task is to check the tails to verify if they actually belong to the assigned cycle or not
    for i in range(len(ww) - 1):
        # if there is a gap between consecutive cycles
        """
        if ww[i + 1][0] - ww[i][-1] > 1:
            for k in range(ww[i][-1] + 1, ww[i + 1][0]):
                t1, t2 = (cycles[k][0][0] - cycles[ww[i][-1]][-1][0]) / np.timedelta64(1, 's'), (
                            cycles[ww[i + 1][0]][0][0] - cycles[k][-1][0]) / np.timedelta64(1, 's')
                if t1 < t2:
                    ww[i] = (ww[i][0], k)
                else:
                    ww[i + 1] = (k, ww[i + 1][1])
                    break
        """
        if ids[ww[i]] == ids[ww[i + 1]]:
            # if duplication
            if ww[i][1] == ww[i + 1][0]:
                t1 = (cycles[ww[i][1]][0][0] - cycles[ww[i][1] - 1][-1][0]) / np.timedelta64(1, 's')
                t2 = (cycles[ww[i][1] + 1][0][0] - cycles[ww[i][1]][-1][0]) / np.timedelta64(1, 's')
                if t1 < t2:
                    ids[(ww[i + 1][0] + 1, ww[i + 1][1])] = ids[ww[i + 1]]
                    ww[i + 1] = (ww[i + 1][0] + 1, ww[i + 1][1])
                else:
                    ids[(ww[i][0], ww[i][1] - 1)] = ids[ww[i]]
                    ww[i] = (ww[i][0], ww[i][1] - 1)

            # check left side first
            if (ww[i][0] != ww[i][1]) and (ww[i + 1][0] != ww[i + 1][1]):
                while True:
                    t1 = (cycles[ww[i][1]][0][0] - cycles[ww[i][1] - 1][-1][0]) / np.timedelta64(1, 's')
                    t2 = (cycles[ww[i + 1][0]][0][0] - cycles[ww[i][1]][-1][0]) / np.timedelta64(1, 's')
                    t_window = (cycles[ww[i][1]][-1][0] - cycles[ww[i][1]][0][0]) / np.timedelta64(1, 's')

                    index1 = patterns[ids[ww[i]]][2].index(ww[i])
                    # print(ww[i], ww[i+1], (patterns[ids[ww[i]]][3][index1] - t_window) / patterns[ids[ww[i]]][3][index1])
                    if t2 < t1 and (patterns[ids[ww[i]]][3][index1] - t_window) / patterns[ids[ww[i]]][3][
                        index1] >= 0.85:
                        #print('a', t1, t2, ww[i][0], ww[i][1] - 1, (ww[i][1], ww[i + 1][1]))
                        ids[(ww[i][0], ww[i][1] - 1)], ids[(ww[i][1], ww[i + 1][1])] = ids[ww[i]], ids[ww[i + 1]]
                        ww[i], ww[i + 1] = (ww[i][0], ww[i][1] - 1), (ww[i][1], ww[i + 1][1])
                        patterns[ids[ww[i]]][2][index1], patterns[ids[ww[i]]][2][index1 + 1] = ww[i], ww[i + 1]
                        patterns[ids[ww[i]]][3][index1] -= t_window
                        patterns[ids[ww[i]]][3][index1 + 1] += t_window
                        #print(patterns[ids[ww[i]]])
                    else:
                        break
                # check right side next
                while True:
                    t1 = (cycles[ww[i + 1][0]][0][0] - cycles[ww[i][1]][-1][0]) / np.timedelta64(1, 's')
                    t2 = (cycles[ww[i + 1][0] + 1][0][0] - cycles[ww[i + 1][0]][-1][0]) / np.timedelta64(1, 's')
                    t_window = (cycles[ww[i + 1][0]][-1][0] - cycles[ww[i + 1][0]][0][0]) / np.timedelta64(1, 's')

                    index2 = patterns[ids[ww[i + 1]]][2].index(ww[i])
                    # print(ww[i], ww[i+1], (patterns[ids[ww[i+1]]][3][index2] - t_window) / patterns[ids[ww[i+1]]][3][index2])
                    if t1 < t2 and (patterns[ids[ww[i + 1]]][3][index2] - t_window) / patterns[ids[ww[i + 1]]][3][
                        index2] >= 0.85:
                        #print('b', (ww[i][0], ww[i + 1][0]), (ww[i + 1][0] + 1, ww[i + 1][1]))
                        ids[(ww[i][0], ww[i + 1][0])], ids[(ww[i + 1][0] + 1, ww[i + 1][1])] = ids[ww[i]], ids[
                            ww[i + 1]]
                        ww[i], ww[i + 1] = (ww[i][0], ww[i + 1][0]), (ww[i + 1][0] + 1, ww[i + 1][1])
                        patterns[ids[ww[i + 1]]][2][index2 - 1], patterns[ids[ww[i + 1]]][2][index2] = ww[i], ww[i + 1]
                        patterns[ids[ww[i + 1]]][3][index2 - 1] += t_window
                        patterns[ids[ww[i + 1]]][3][index2] -= t_window
                    else:
                        break
    ww_ = []
    for i in ww:
        if i not in ww_:
            ww_ += [i]
    return ww_


def cycle_length(intervals):
    # output = intervals[0][1] - intervals[0][0] + 1
    output = 0
    for i in range(0, len(intervals)):
        if intervals[i][1] != -1:
            output += intervals[i][1] - intervals[i][0] + 1 - [0,1][intervals[i][0] == intervals[i-1][1]]
    return output


def best_cycle(patterns, quantity):
    # temp = [(i, patterns[i][0], np.mean(patterns[i][1]), gap_size(patterns[i][2]), abs(quantity - len(patterns[i][2])), np.std(patterns[i][-2]), cycle_length(patterns[i][2])) for i in patterns.keys() if len(patterns[i][2]) > 1]
    temp = [(i, patterns[i][0], min([j for j in patterns[i][1] if not np.isnan(j)]), gap_size(patterns[i][2]), abs(quantity - len([k for k in patterns[i][2] if k[1] != -1])), np.std(patterns[i][-2]), cycle_length(patterns[i][2])) for i in patterns.keys() if len(patterns[i][2]) > 1]
    # order = number of active phase used, diff b/w recorded quantity and # of detected cycles,
    #         number of not used active phase, average of distance difference
    if quantity == -1: # if no SO mapping with quantity info is available
        temp = sorted(temp, key=lambda xx: [-xx[-1], xx[3], xx[2]])
    else:
        temp = sorted(temp, key=lambda xx: [-xx[-1], xx[4], xx[3], xx[2]])
    #print(temp[:3])

    output, exclude, c, diff, dist = [], set(), 0, float('inf'), float('inf')

    visited = set()
    for i in range(len(temp)):
        if i not in visited:
            visited.add(temp[i][0])
            output_, exclude_, c_, d_, diff_ = [temp[i][0]], set(patterns[temp[i][0]][2]), temp[i][-1], temp[i][2], temp[i][4]
            temp_ = [i for i in temp if overlapped(exclude_, patterns[i[0]][2]) and i[0] not in visited]
            while len(temp_) > 0:
                if quantity == -1: # if no SO mapping with quantity info is available
                    temp = sorted(temp, key=lambda xx: [-xx[-1], xx[3], xx[2]])
                else: # if SO mapping is available, let's look for the one with similar quantity
                    temp = sorted(temp, key=lambda xx: [-xx[-1], xx[4], xx[3], xx[2]])
                output_ += [temp_[0][0]]
                c_ += temp_[0][-1]
                d_ += temp_[0][2]
                diff_ += temp_[0][4]
                exclude_ |= set(patterns[temp_[0][0]][2])
                # the rest
                temp_ = [i for i in temp_ if overlapped(exclude_, patterns[i[0]][2]) and i[0] not in visited]
            #print(c_, c, output_, d_, dist)
            if c_ > c:
                output, exclude, c, dist, diff = output_, exclude_, c_, d_, diff_
            elif c_ == c and d_ < dist and diff_ < diff_:
                output, exclude, c, dist, diff = output_, exclude_, c_, d_, diff_
    complete_cycle = []
    for i in output:
        complete_cycle += patterns[i][2]
    return output, complete_cycle


def closer(index, cycles):
    if 0 < index + 1 < len(cycles):
        t0, t1 = cycles[index-1][-1][0], cycles[index][0][0]
        t2, t3 = cycles[index][-1][0], cycles[index+1][0][0]
        # which one is closer!
        if (t1 - t0) / np.timedelta64(1, 's') > (t3 - t2) / np.timedelta64(1, 's'):
            return False
    return True


def initial_phase(timelapse, index, cycles):
    x1, t1 = [], 0
    i, t = 0, max(min(timelapse // 8, 600), 480)
    while t1 <= t and index+i < len(cycles):
        t1 += time_diff_calculator(index+i, cycles)
        x1 += [x_[1] for x_ in cycles[index+i]]
        i += 1
    return x1


def end_phase(timelapse, start, end, cycles, keep=0):
    x1, t1 = [], 0
    if keep == 0:
        t = max(min(timelapse // 8, 600), 480)
        while t1 <= t and end > start:
            t1 += time_diff_calculator(end, cycles)
            x1 += [x_[1] for x_ in cycles[end]][::-1]
            end -= 1
    else:
        while end >= start and len(x1) < keep:
            x1 += [x_[1] for x_ in cycles[end]][::-1]
            end -= 1
        x1 = x1[:keep]
    return x1


def compare_phase(x1, x2, t='start'):
    l1, l2 = len(x1), len(x2)
    l_limit = int(min(l1, l2)*1.2)
    #init_dist = (compute_dtw(np.array(x1[:min(l1, l2)]), np.array(x2[:min(l1, l2)])) + compute_euclidean(np.array(x1[:min(l1, l2)]), np.array(x2[:min(l1, l2)]))) / (2*min(l1, l2) / max(l1, l2))
    #if t == 'start':
    #    init_dist = compute_euclidean(np.array(x1[:min(l1, l2)]), np.array(x2[:min(l1, l2)]))
    #else:
    #print(compute_dtw(np.array(x1[:l_limit]), np.array(x2[:l_limit])), compute_euclidean(np.array(x1[:min(l1, l2)]), np.array(x2[:min(l1, l2)])))
    init_dist = 0.7*compute_dtw(np.array(x1[:l_limit]), np.array(x2[:l_limit])) + 0.3*compute_euclidean(np.array(x1[:min(l1, l2)]), np.array(x2[:min(l1, l2)]))
    #init_dist = (compute_dtw(np.array(x1[:l_limit]), np.array(x2[:l_limit])) + compute_euclidean(np.array(x1[:min(l1, l2)]), np.array(x2[:min(l1, l2)]))) / (2*min(l1, l2) / max(l1, l2))
    return init_dist


def start_phase_matching(index, key, cycles, first_phase, patterns, cycle_parameter):
    exclude = []  # if the time between previous and current active phase is within certain amount time interval, don't consider it as an candidate
    for i in range(index + 1, len(cycles)):
        if (cycles[i][0][0] - cycles[i - 1][-1][0]) / np.timedelta64(1, 's') <= cycle_parameter['cycle_gap']:
            exclude += [i]

    dist_temp = []
    for i in range(index, len(cycles)):
        if i not in exclude:
            x1, x2 = first_phase[key][-1], initial_phase(patterns[key][0], i, cycles)
            # plt.plot(x1)
            # plt.plot(x2)
            init_dist = compare_phase(x1, x2)
            # plt.title(str(index+i) + " " + str(init_dist))
            # plt.show()
            dist_temp += [init_dist]
        else:
            dist_temp += [float('inf')]

    if cycle_parameter['upper_limit'] < 60:
        candidate_point = [i for i in range(len(dist_temp)) if dist_temp[i] <= cycle_parameter['min_dist'] * 2]
    else:
        dist_temp = [float('inf')] + dist_temp + [float('inf')]
        # print(dist_temp)
        candidate_point = []
        for i in range(1, len(dist_temp) - 1):
            if dist_temp[i - 1] > dist_temp[i] and dist_temp[i] < dist_temp[i + 1] and dist_temp[i] <= cycle_parameter['min_dist']:
                candidate_point += [i - 1]
        rest = [i for i in sorted(enumerate(dist_temp[1:-1]), key=lambda x: x[1]) if i[0] not in candidate_point]
        candidate_point += [i for i, _ in rest[:6]]
        # candidate_point += [i for i, _ in rest[:len(cycles)//2 if len(cycles)%2!=1 else len(cycles)//2+1]]
    return [i + index for i in sorted(candidate_point)]


def pattern_detector(cycles, cycle_parameter, machine, order_number):
    patterns, candidates, pattern_index, start_time, first_phase, last_phase, previous_id = pattern_history_parse(machine, order_number)
    m = 10
    relaxation_factor = 3

    # initialization if the history exists to start
    previous_match = {}
    for key in first_phase.keys():
        previous_match[key] = start_phase_matching(0, key, cycles, first_phase, patterns, cycle_parameter)

    for i in range(len(cycles)):
        timelapse, xx, yy = 0, [], []
        for k in range(cycle_parameter['gap']):
            if i + k < len(cycles) and (k > 0 and (cycles[i + k][0][0] - cycles[i + k - 1][-1][0]) / np.timedelta64(1, 's') >= (cycle_parameter['lower_limit'] + cycle_parameter['upper_limit']) * 30 * m):
                break
            if i + k < len(cycles) and (k == 0 or (k > 0 and (cycles[i + k][0][0] - cycles[i + k - 1][-1][0]) / np.timedelta64(1, 's') < (cycle_parameter['lower_limit'] + cycle_parameter['upper_limit']) * 30 * m)):
                current = cycles[i + k]
                timelapse += time_diff_calculator(i + k, cycles)
                xx += [x_[0] for x_ in current]
                yy += [x_[1] for x_ in current]
                if cycle_parameter['lower_limit'] * 60 <= timelapse <= cycle_parameter['upper_limit'] * 60:
                    try:
                        sm1 = poly_smooth(xx, yy, cycle_parameter)  # polyfit
                    except:
                        print('not available', i, i + k)
                        sm1 = []

                    notFound = True
                    for key in patterns.keys():
                        timelapse_, score_, i_, tt, _ = patterns[key]
                        # print(key, previous_match[key])
                        include = (i in previous_match[key] or cycle_parameter['lower_limit'] <= 30)
                        # print(key, i, i+k, previous_match[key], timelapse, timelapse_, len(sm1), include, len(i_) > 1 or (len(i_) == 1 and i_[-1][1] < i))
                        if 1 - (cycle_parameter['tolerance'] / 100) <= min(timelapse, timelapse_) / max(timelapse, timelapse_) and (len(i_) > 1 or (len(i_) == 1 and i_[-1][1] < i)) and sm1 != [] and include:
                            sm2 = candidates[key][[-1, -2][i_[-1][0] == i]]
                            dist = compute_euclidean(sm1, sm2)
                            initial_ = initial_phase(timelapse, i, cycles)
                            last_ = end_phase(timelapse_, i, i + k, cycles, len(last_phase[key][-1]))
                            # 마지막 지점 다음 지점이 시작점이거나 or 끝지점이어야한다
                            if dist <= cycle_parameter['min_dist'] and (i+k == len(cycles)-1 or (i+k+1 < len(cycles) and i+k+1 in previous_match[key])) or config.company in {'jm'}:
                                case_ = (cycle_parameter['upper_limit'] < 20 and i > i_[-1][1]) or (cycle_parameter['upper_limit'] >= 20 and i >= i_[-1][1])
                                if case_ and (i - i_[-1][1] < cycle_parameter['upper_gap'] or i_[-1][1] == -1) and ((i_[-1][0] == i_[-1][1] and i_[-1][1] < i) or i_[-1][1] < i):  # no overlapping
                                    patterns[key][1] += [dist]
                                    patterns[key][2] += [(i, i + k)]
                                    patterns[key][3] += [timelapse]
                                    patterns[key][4] += [k + [0, 1][i != i_[-1][1]]]
                                    patterns[key][0] = np.mean(patterns[key][3][:-1])
                                    candidates[key] += [sm1]
                                    first_phase[key] += [initial_]
                                    last_phase[key] += [last_]
                                    notFound = False
                                elif i <= i_[-1][1] and score_ != [] and i_[-1][0] == i:  # overlapping, so let's choose the better one
                                    last_1 = np.median([compare_phase(last_phase[key][ll], last_phase[key][-1], 'last') for ll in range(len(last_phase[key]) - 1)])
                                    last_2 = np.median([compare_phase(last_phase[key][ll], last_, 'last') for ll in range(len(last_phase[key]) - 1)])
                                    if last_2 < last_1:
                                        # print('b')
                                        patterns[key][1][-1] = dist
                                        patterns[key][2][-1] = (i, i + k)
                                        patterns[key][3][-1] = timelapse
                                        patterns[key][4][-1] = k + [0, 1][i != i_[-2][1]]
                                        patterns[key][0] = np.mean(patterns[key][3][:-1])
                                        candidates[key][-1] = sm1
                                        last_phase[key][-1] = last_
                                        notFound = False
                    if notFound and sm1 != []:
                        patterns[pattern_index] = [timelapse, [], [(i, i + k)], [timelapse], [k + 1]]  # timelapse, distance score, cycle index
                        candidates[pattern_index] = [sm1]
                        first_phase[pattern_index] = [initial_phase(timelapse, i, cycles)]
                        last_phase[pattern_index] = [end_phase(timelapse, i, i + k, cycles)]

                        previous_match[pattern_index] = start_phase_matching(i + k + 1, pattern_index, cycles,
                                                                             first_phase, patterns, cycle_parameter) + [i]
                        pattern_index += 1
                elif timelapse > cycle_parameter['upper_limit'] * 60:
                    break
    return patterns, candidates, first_phase, last_phase


def build_cycles(complete_cycles, cycles, machine, order_number, cycle_parameter):
    """
    If best candidate suggested does not provide clear solution, user can suggest another candidate manually

    :param complete_cycles:
    :param cycles:
    :param machine:
    :param order_number:
    :return:
    """
    patterns, candidates, pattern_index, start_time, first_phase, last_phase, previous_id = pattern_history_parse(
        machine, order_number)

    output, cycle_output = [], []
    for complete_cycle in complete_cycles:
        cycle_output += complete_cycle
        index = 0
        for start, stop in complete_cycle:
            timelapse, xx, yy = 0, [], []
            for i in range(start, stop + 1):
                current = cycles[i]
                timelapse += time_diff_calculator(i, cycles)
                xx += [x_[0] for x_ in current]
                yy += [x_[1] for x_ in current]

            sm1 = poly_smooth(xx, yy, cycle_parameter)
            if index == 0:
                candidates[pattern_index] = [sm1]
                first_phase[pattern_index] = [initial_phase(timelapse, start, cycles)]
                last_phase[pattern_index] = [end_phase(timelapse, start, stop, cycles)]
                patterns[pattern_index] = [timelapse, [], [(start, stop)], [timelapse], [stop - start + 1]]
            else:
                sm2 = candidates[pattern_index][-1]
                dist = compute_euclidean(sm1, sm2)
                initial_ = initial_phase(timelapse, start, cycles)
                last_ = end_phase(timelapse, start, stop, cycles, len(last_phase[pattern_index][-1]))
                patterns[pattern_index][1] += [dist]
                patterns[pattern_index][2] += [(start, stop)]
                patterns[pattern_index][3] += [timelapse]
                patterns[pattern_index][4] += [stop - start]
                patterns[pattern_index][0] = np.mean(patterns[pattern_index][3][:-1])
            index += 1
        output += [pattern_index]
        pattern_index += 1
    return patterns, candidates, first_phase, last_phase, output, cycle_output
