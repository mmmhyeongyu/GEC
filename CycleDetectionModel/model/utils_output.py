from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import datetime as dt
import time

from model.utils_cycle import *
from model.utils_function import *

# plotting parameter
plot_param = {
    "fill_y_low_range": [0, 0],
    "fill_y_high_range": [4000, 4000],
    "fill_color": "C0",
    "fill_alpha": 0.3
}
plot_color = {i: j for i, j in zip(['A', 'B', 'C', 'D', 'E', 'F', 'G']  , ['y', 'b', 'k', 'c', 'g', 'r', 'm'])}

def cycle_details(final, cycles):
    intervals = []  # complete cycle start/end
    cycle_history = []  # cycle details
    for ii, jj in final[-1]:
        intervals += [[cycles[ii][0][0], cycles[jj][-1][0]]]
        cycle_history += [[]]
        for k in range(ii, jj + 1):
            cycle_history[-1] += [[cycles[k][0][0], cycles[k][-1][0]]]
    return intervals, cycle_history


def mergeIntervals(a, b):
    i = j = 0
    m, n = len(a), len(b)
    output = []
    while i < m or j < n:
        if i == m:
            output += [b[j]]
            j += 1
        elif j == n:
            output += [a[i]]
            i += 1
        elif a[i][0] > b[j][1]:  # no overlapping and a is bigger
            output += [b[j]]
            j += 1
        elif b[j][0] > a[i][1]:  # no overlapping and b is bigger
            output += [a[i]]
            i += 1
        else:  # overlapping somewhere
            output += [[min(a[i][0], b[j][0]), max(a[i][1], b[j][1])]]
            i += 1
            j += 1

    # now check if we need to potentially merge within current output
    output_ = [output[0]]
    for i in range(1, len(output)):
        if output_[-1][0] <= output[i][0] <= output_[-1][1]:
            output_[-1] = [output_[-1][0], max(output_[-1][1], output[i][1])]
        else:
            output_ += [output[i]]
    return output_


def door_sensor_interval(door_sensors, df):
    door_sensor = {}
    for door in door_sensors:
        temp = df[df["mac"] == door]
        # print(door, df.shape, temp.shape)
        door_sensor[door] = []

        history, previous = [], 0
        for row in temp[["timestamp", "value"]].values:
            # print(row[0], row[1], previous)
            if row[1] == 1 and row[1] != previous:
                history = [row[0]]
            elif row[1] == -1 and previous == 1:
                door_sensor[door] += [history + [row[0]]]
                history = []
            previous = row[1]
    door_merged = []
    if len(door_sensors) == 2:
        door_merged = mergeIntervals(door_sensor[door_sensors[0]], door_sensor[door_sensors[1]])
    return door_merged


def labelling(door, cycle, cycle_history):
    """
    A: Appropriate door open to put material inside
    B: Appropriate door open to take product outside
    C: Door opening during cycle
    D: Machine running
    E: Door/machine cycle conflict?
    F: idle while cycle in process
    G: idle when cycle is being initiated or complete
    """
    label = {i: [] for i in "ABCDEFG"}
    j, n2 = 0, len(cycle)
    if door == []:
        while j < n2:
            label["D"] += [cycle_history[j][0]]
            for k in range(1, len(cycle_history[j])):
                label["F"] += [[cycle_history[j][k - 1][-1], cycle_history[j][k][0]]]
                label["D"] += [cycle_history[j][k]]
            j += 1
    elif door != []:
        i, n1 = 0, len(door)
        while i < n1 or j < n2:
            # print(i, j, n1, n2)
            if i == n1:  # no more door in action
                if j < n2:  # but machine active
                    # let's see if there was any idle time during the cycle
                    label["D"] += [cycle_history[j][0]]
                    for k in range(1, len(cycle_history[j])):
                        label["F"] += [[cycle_history[j][k - 1][-1], cycle_history[j][k][0]]]
                        label["D"] += [cycle_history[j][k]]
                    j += 1
            elif j == n2:  # no more cycle in action
                label["A"] += [door[i]]
                i += 1
            elif door[i][1] <= cycle[j][0]:
                if (cycle[j][0] - door[i][1]).total_seconds() > 30:
                    i += 1
                else:
                    label["A"] += [door[i]]
                    label["G"] += [[door[i][1], cycle[j][0]]]
                    i += 1
            elif door[i][0] <= cycle[j][0] <= door[i][1]:  # # door open to insert material
                label["A"] += [door[i]]
                i += 1
            elif cycle[j][0] <= door[i][0] <= door[i][1] <= cycle[j][1]:  # door open during cycle
                label["C"] += [door[i]]
                i += 1
            elif door[i][0] <= cycle[j][1] <= door[i][1]:  # # door open to take product out?
                label["B"] += [door[i]]
                i += 1
                # let's see if there was any idle time during the cycle
                label["D"] += [cycle_history[j][0]]
                for k in range(1, len(cycle_history[j])):
                    label["F"] += [[cycle_history[j][k - 1][-1], cycle_history[j][k][0]]]
                    label["D"] += [cycle_history[j][k]]
                j += 1  # cycle is complete too!
            elif cycle[j][1] <= door[i][0]:  # door open to take product out, but there is idle time
                label["G"] += [[cycle[j][1], door[i][0]]]
                label["B"] += [door[i]]
                i += 1
                # let's see if there was any idle time during the cycle
                label["D"] += [cycle_history[j][0]]
                for k in range(1, len(cycle_history[j])):
                    label["F"] += [[cycle_history[j][k - 1][-1], cycle_history[j][k][0]]]
                    label["D"] += [cycle_history[j][k]]
                j += 1  # cycle is complete too!
    return label


def final_output_visualization(final_label, f, df_set, mac_main):
    label_ = final_label.copy()
    bb = 24
    start = f[0]
    index_ = [0] * len(label_.keys())
    for _ in range(1):
        plt.figure(figsize=(15, 40))
        for i in range(bb):
            ax = plt.subplot(bb, 1, i + 1)
            end = start + timedelta(hours=24 // bb)

            df_subset = df_set[(df_set["timestamp"] >= start) & (df_set["timestamp"] < end)]
            # plotting values
            for mac in [mac_main]:
                df_subset_ = df_subset[df_subset["mac"] == mac]
                ax.plot(df_subset_["timestamp"], df_subset_["value"])
            # adding shade if label 1
            for index, l in enumerate(list(label_.keys())):
                if l != "E":
                    left = index_[index]
                    # print(l, left, len(label_[l]), label_[l][left], start, end)
                    while left < len(label_[l]):
                        s, e = label_[l][left]
                        # print(label_[l][left])
                        # s, e = datetime.strptime(s, '%Y-%m-%d %H:%M:%S'), datetime.strptime(e, '%Y-%m-%d %H:%M:%S')
                        if start <= s < end:
                            # print(l, s, min(e, end))
                            ax.fill_between([s, min(e, end)],
                                            plot_param["fill_y_low_range"],
                                            plot_param["fill_y_high_range"],
                                            color=plot_color[l],
                                            alpha=plot_param["fill_alpha"])
                            if e < end:
                                left += 1
                            else:
                                label_[l][left][0] = end
                        else:
                            index_[index] = left
                            break
            ax.set_xlim([start, end])
            ax.set_ylim([0, 10000])
            ax.grid(axis='x')
            start += timedelta(hours=24 // bb)
        plt.show()


def so_cycle_output(complete_cycle, cycles, result, machine, material_, order_number_, gop_, routing_revision_, quantity_):

    # get sensorMetaData
    factory_id, process_id, equipment_id = sensor_meta(machine)

    active_, inactive_, total_, cycle_number_ = [], [], [], 1
    log, idle = [], []
    complete_cycle = [cy for cy in complete_cycle if cy[1] != -1]
    for cy in complete_cycle:
        start_cycle, end_cycle = cycles[cy[0]][0][0], cycles[cy[1]][-1][0]
        active, inactive = 0, 0
        for i in range(cy[0], cy[1] + 1):
            active += (cycles[i][-1][0] - cycles[i][0][0]) / np.timedelta64(1, 's')
            log += [
                [(cycles[i][0][0], cycles[i][-1][0]), (cycles[i][-1][0] - cycles[i][0][0]) / np.timedelta64(1, 's')]]
            if i != cy[1]:
                inactive += (cycles[i + 1][0][0] - cycles[i][-1][0]) / np.timedelta64(1, 's')
                idle += [[(cycles[i][-1][0], cycles[i + 1][0][0]),
                          (cycles[i + 1][0][0] - cycles[i][-1][0]) / np.timedelta64(1, 's')]]
        active_ += [active]
        inactive_ += [inactive]
        total_ += [active + inactive]

        result = result.append({'factory_id':factory_id, 'process_id':process_id, 'equipment_id':equipment_id, 'equipment_name':machine,
                                    'material':material_, 'routing_revision':routing_revision_, 'so':order_number_, 'gop':gop_, 'qty':quantity_, 'cycle_number':cycle_number_,
                                    'start_time':start_cycle, 'stop_time':end_cycle,
                                    'spindle': round(active/3600,3), 'idle_inter': round(inactive/3600,3), 'total': round((active+inactive)/3600,3)}, ignore_index=True)
        cycle_number_ += 1
    
    return result


def daily_cycle_output(complete_cycle, cycles, start_, machine, mac_main):
    box_style = {'facecolor': 'w', 'edgecolor': 'w', 'boxstyle': 'round', 'alpha': 0.5}

    # read JM 진명 specific meta data from json
    with open("JM_meta.json", "r") as fin:
        jm_meta = json.load(fin)

    # we need to generate 3 output based on: https://gentleenergycorp.atlassian.net/wiki/spaces/DS001/pages/379944961/Output+Format
    daily_summary = pd.DataFrame(columns=['process_id', 'location', 'equipment_id', 'name',
                                          'num', 'model', 'timestamp', 'run_day_load', 'run_day_up', 'run_day_down',
                                          'run_day_per', 'run_night_load', 'run_night_up', 'run_night_down',
                                          'run_night_per', 'product_day_plan', 'product_day_real', 'product_day_per',
                                          'product_night_plan', 'product_night_real', 'product_night_per'])

    hourly_equipment_summary = pd.DataFrame(columns=['process_id', 'location', 'equipment_id', 'name',
                                                     'shift', 'timestamp', 'start', 'end', 'run_load',
                                                     'run_up', 'run_down', 'run_per',
                                                     'product_plan', 'product_real', 'product_per'])

    hourly_summary = pd.DataFrame(columns=['factory_id', 'machine_id', 'mac', 'sensor_type',
                                           'timestamp', 'event_type', 'event_count', 'frame_min',
                                           'uptime_min', 'active_min', 'sensor_id'])

    jm_meta_ = jm_meta[machine]
    process_id, location, equipment_id = jm_meta_['process_id'], jm_meta_['location'], jm_meta_['equipment_id']
    num_, model_, factory_id, sensor_type = jm_meta_['num'], jm_meta_['model'], 18, '진동'

    # let's break down complete cycles
    summary_ = {}
    for i in range(24):
        tt = str(pd.to_datetime(start_) + pd.DateOffset(minutes=60 * i))
        summary_[int(str(tt)[11:13])] = [0, 0, tt]  # active minutes, number of completed cycles

    # use complete cycle to obtain number of cycles information
    for i in complete_cycle:
        start_time, end_time = cycles[i[0]][0][0], cycles[i[1]][-1][0]
        start_hour, end_hour = int(str(start_time)[11:13]), int(str(end_time)[11:13])
        # print(i, start_time, end_time, round((end_time - start_time) // np.timedelta64(1, 's')  / 60, 2))
        if start_hour == end_hour:
            summary_[start_hour][1] += 1
        else:
            # process the start_hour first
            end_time_ = pd.to_datetime(str(start_time)[:13]) + pd.DateOffset(minutes=60)
            summary_[start_hour][1] += 1

    # use active phase (cycles) to fill active times
    for i in range(len(cycles)):
        start_time, end_time = cycles[i][0][0], cycles[i][-1][0]
        start_hour, end_hour = int(str(start_time)[11:13]), int(str(end_time)[11:13])
        # print(i, start_time, end_time, round((end_time - start_time) // np.timedelta64(1, 's')  / 60, 2))
        if start_hour == end_hour:
            summary_[start_hour][0] += round((end_time - start_time) // np.timedelta64(1, 's') / 60, 2)
        else:
            # process the start_hour first
            end_time_ = pd.to_datetime(str(start_time)[:13]) + pd.DateOffset(minutes=60)
            summary_[start_hour][0] += round((end_time_ - start_time) // np.timedelta64(1, 's') / 60, 2)
            start_time = end_time_
            while True:
                start_hour = int(str(start_time)[11:13])
                if start_hour == end_hour:
                    summary_[start_hour][0] += round((end_time - start_time) // np.timedelta64(1, 's') / 60, 2)
                    break
                start_time += pd.DateOffset(minutes=60)

    active, load, numCompleteCycles = [0, 0], [480, set()], [0, 0]

    for i in list(range(8, 24)) + list(range(0, 8)):
        # daily summary
        if 17 <= i < 20 and summary_[i][0] > 0:
            load[0] = 630
        if 8 <= i < 20:  # daytime
            active[0] += summary_[i][0]
            numCompleteCycles[0] += summary_[i][1]
        else:  # night time
            active[1] += summary_[i][0]
            numCompleteCycles[1] += summary_[i][1]
            if summary_[i][0] > 0:
                load[1].add(i)

        # hourly summary
        shift_ = ['주간조', '야간조'][i < 8 or i >= 20]
        run_load_ = 60
        if i in [12, 0]:
            run_load_ = 0
        elif i in [17, 5]:
            run_load_ = 30
        timestamp_ = summary_[i][2]

        hourly_equipment_summary = hourly_equipment_summary.append({
            'process_id': process_id, 'location': location, 'equipment_id': equipment_id, 'name': machine,
            'shift': shift_, 'timestamp': timestamp_, 'start': str(i) + ":00", "end": str((i + 1) % 24) + ":00",
            'run_load': run_load_, 'run_up': summary_[i][0], 'run_down': max(0, run_load_ - summary_[i][0]),
            'run_per': int(summary_[i][0] / run_load_ * 100) if run_load_ > 0 else 0, 'product_plan': "",
            'product_real': summary_[i][1],
            'product_per': ""
        }, ignore_index=True)

        # we need to convert timestamp to utc for the following table
        timestamp_ = str(pd.to_datetime(timestamp_) - pd.DateOffset(hours=9))
        hourly_summary = hourly_summary.append({
            'factory_id': factory_id, 'machine_id': equipment_id, 'mac': "",
            'sensor_type': "",
            'timestamp': timestamp_, 'event_type': 'prod_cnt', 'event_count': summary_[i][1],
            'frame_min': 60, 'uptime_min': 60, 'active_min': summary_[i][0], 'sensor_id': ""
        }, ignore_index=True)

        uptime_min = 60
        if summary_[i][0] == 0:
            uptime_min = 0
        elif summary_[i][0] <= 20:
            uptime_min = 40
        hourly_summary = hourly_summary.append({
            'factory_id': factory_id, 'machine_id': equipment_id, 'mac': "",
            'sensor_type': "",
            'timestamp': timestamp_, 'event_type': 'action', 'event_count': 0,
            'frame_min': 60, 'uptime_min': uptime_min, 'active_min': summary_[i][0], 'sensor_id': ""
        }, ignore_index=True)

    load[1] = len(load[1]) * 60
    daily_summary = daily_summary.append(
        {'process_id': process_id, 'location': location, 'equipment_id': equipment_id, 'name': machine,
         'num': num_, 'model': model_, 'timestamp': start_,
         'run_day_load': load[0], 'run_day_up': active[0], 'run_day_down': load[0] - active[0],
         'run_day_per': int(active[0] / load[0] * 100) if load[0] > 0 else 0,
         'run_night_load': load[1], 'run_night_up': active[1], 'run_night_down': load[1] - active[1],
         'run_night_per': int(active[1] / load[1] * 100) if load[1] > 0 else 0,
         'product_day_plan': "", 'product_day_real': numCompleteCycles[0], 'product_day_per': "",
         'product_night_plan': "", 'product_night_real': numCompleteCycles[1], 'product_night_per': ""
         }, ignore_index=True)
    return daily_summary, hourly_equipment_summary, hourly_summary

def so_pattern_output(output, patterns, candidates, machine, complete_cycle, times, ss, cycles, first_phase, last_phase, material_, order_number, cycle_parameter):
    # if complete_cycle == [] or (quantity_ < 5 and start_time == "") or (quantity_ == 1 and cycle_length([xx for xx in complete_cycle if xx[1] != -1]) / len(cycles) <= 0.8):
    if complete_cycle != []:
        print("##########################################################################################")
        print("Current best candidates plot")
        store_cycle_output_and_plot(output, patterns, candidates, machine, complete_cycle, times, ss, cycles,
                                    first_phase, last_phase, material_, order_number, False)
    print("##########################################################################################")
    print("Current best candidates: ",
          "|".join([":".join([str(xx[0]) + "," + str(xx[1]) for xx in patterns[cc][2]]) for cc in output]))
    print("Size of active phases: ", len(cycles))
    print("##########################################################################################")
    best_candidates = "|".join([":".join([str(xx[0]) + "," + str(xx[1]) for xx in patterns[cc][2]]) for cc in output])

    while True:
        print('\r\n')
        print('_______________________________________________________________________________________')
        # print("1. Press enter with empty blank if no cycle is found...")
        # print("2. Copy/Paste the best candidates from above if it looks good...")
        print("1. 'Y': Select best candidates or Press enter with empty blank")
        print("2. 'N': best candidates looks good...")

        # flag indicating if the cycle was built manually or not
        flag = False  # false if automatic; true otherwise (manual)

        while True:
            try:
                time.sleep(1)
                proceed = input("Do we need to change best candidates? (Y/N):")
                proceed = "Y" if best_candidates == "" else proceed
                if proceed == "N":
                    candidates_ = best_candidates
                    candidates_ = [[tuple([int(x) for x in candi.split(",")]) for candi in candi_.split(":")] for candi_ in
                                candidates_.split("|")]
                    break
                elif proceed == "Y":
                    time.sleep(1)
                    candidates_ = input(
                        "Complete set of indices of cycles (, as delimiter for indices and : as delimiter for cycles = ")
                    if candidates_.strip() == "":
                        complete_cycle = []
                        break
                    else:
                        candidates_ = [[tuple([int(x) for x in candi.split(",")]) for candi in candi_.split(":")] for candi_ in
                                candidates_.split("|")]
                        break
                else: continue
            except:
                print("Possibly inappropriate charater was added? Please fill out the cycle candidates again...")

        output, candidate_cycles = [], []
        for candi in candidates_:
            for i in patterns.keys():
                if patterns[i][2] == candi:
                    candidate_cycles += patterns[i][2]
                    output += [i]
                    break
        if output != []:
            store_cycle_output_and_plot(output, patterns, candidates, machine, sorted(candidate_cycles), times, ss,
                                        cycles, first_phase, last_phase, material_, order_number, False)
            time.sleep(1)
            check = input("Y if looks fine; any input otherwise: ")
            if check == "Y":
                complete_cycle = sorted(candidate_cycles)
                break
        elif output == [] and candidate_cycles == [] and candidates_ == []:
            complete_cycle = []
            print("No cycle was found, so skipping the current SO")
            break
        else:
            print("Building new cycle patterns based on user input")
            flag = True
            time.sleep(2)
            patterns, candidates, first_phase, last_phase, output, complete_cycle = build_cycles(candidates_, cycles,
                                                                                                 machine, order_number, cycle_parameter)
            
            store_cycle_output_and_plot(output, patterns, candidates, machine, sorted(complete_cycle), times, ss,
                                        cycles, first_phase, last_phase, material_, order_number, False)
            complete_cycle = []
            for i in output:
                complete_cycle += patterns[i][2]
            print("Final complete cycle: ", complete_cycle)
            time.sleep(2)
            check = input("Y if looks fine; any input otherwise: ")
            if check == "Y":
                # complete_cycle = sorted(candidate_cycles)
                # print('now? ', patterns)
                print('now? ', output)
                break
    return output, patterns, candidates, machine, complete_cycle, first_phase, last_phase, flag

def sensor_meta(machine):
    with open('./parameter/equipment.json', 'r') as fin:
        sensor_meta = json.load(fin)
    factory_id = sensor_meta[machine]['factory_id']
    process_id = sensor_meta[machine]['process_id']
    equipment_id = sensor_meta[machine]['equipment_id']
    return factory_id, process_id, equipment_id

def params(machine):
    with open('./parameter/equipment_properties.json', 'r') as fin:
        parameter = json.load(fin)
    gap = parameter[machine]['gap']
    threshold = parameter[machine]['threshold']
    min_threshold = parameter[machine]['min_threshold_value']
    smooth_factor = parameter[machine]['smooth_factor']
    step = parameter[machine]['step']
    upper_limit = parameter[machine]['upper']
    # poly_order = parameter[machine]['order']
    mac_main = parameter[machine]['mac']
    # door_sensors = parameter[machine]['door_sensor']
    window = parameter[machine]['window']
    max_cycle = parameter[machine]['max_cycle']
    idle_value = parameter[machine]['idle_value']
    return parameter, gap, threshold, min_threshold, smooth_factor, step, upper_limit, mac_main, window, max_cycle, idle_value

def quantityRedefine(machine, quantity):
    if machine == 'MC18200 #1' or machine == 'MP2150':
        quantity = quantity*2
    elif machine == 'CMI 3S #8':
        quantity = quantity*3
    return quantity

def params_redefine(temp, machine, so_range):
    parameter, gap, threshold, min_threshold, smooth_factor, step, upper_limit, mac_main, window, max_cycle, idle_value = params(machine)

    print(f'''
    Current Values
    min_threshold = {min_threshold}
    smooth_factor = {smooth_factor}
    step = {step}
    upper = {upper_limit}
    window = {window}
    ''')
    
    time.sleep(1)
    min_threshold = int(input("min_threshold (threshold value to define inactive phase (default: 0)) = "))
    smooth_factor = int(input("smooth_factor (set higher value if too noisy; lower otherwise) = "))
    step = int(input("step (set higher value if too noisy; lower otherwise) = "))
    upper_limit = int(input("upper (set higher value if current value catches too many active phase; lower otherwise) = "))
    window = int(input("Window (set lower value if noisy with significant value shift; higher otherwise) = "))

    times_, y_ = cleaning(temp, smooth_factor, min_threshold)
    smooth_ = smoothing(y_, step)
    times, ss, inactive_cycles_ = inactive_phase_cleaning_normalization(smooth_, times_, upper_limit, window, so_range)

    print(f'''
    Current Values
    threshold = {threshold}
    upper = {upper_limit}
    max_cycle = {max_cycle}
    gap = {gap}
    idle_value = {idle_value}
    ''')
    
    time.sleep(1)
    threshold = float(input("Threshold (set higher value if each active cycle needs to be smaller; lower otherwise) = "))
    upper_limit = int(input("upper (set higher value if current value catches too many active phase; lower otherwise) = "))
    max_cycle = int(input("Max_cycle (set higher value if each active cycle needs to be larger; lower otherwise) = "))
    gap = int(input("Gap (allowable gap between consecutive active phases in seconds) = "))
    idle_value = float(input("Idle value (value defines the active/inactive phase (any value less than idle_value is considered to be inactive)) = "))

    cycles_ = active_phase(ss, times, threshold, upper_limit, max_cycle * 60, gap, idle_value)
    cycles = cycle_generate(cycles_, times, ss, inactive_cycles_, so_range)
    
    parameter[machine]['gap'] = gap
    parameter[machine]['threshold'] = threshold
    parameter[machine]['min_threshold_value'] = min_threshold
    parameter[machine]['smooth_factor'] = smooth_factor
    parameter[machine]['step'] = step
    parameter[machine]['upper'] = upper_limit
    parameter[machine]['window'] = window
    parameter[machine]['max_cycle'] = max_cycle
    parameter[machine]['idle_value'] = idle_value

    with open('./parameter/equipment_properties.json', 'w') as fout:
        json.dump(parameter, fout)
    return times, ss, cycles

def db_insert(df, tbl=''):
    insert_df = df.copy().fillna(0)
    fields = ', '.join(insert_df.columns)
    values = ', '.join(['%s'] * len(insert_df.columns))
    operation = f"INSERT INTO {tbl} ({fields}) VALUES ({values});"
    records = [tuple(row) for row in list(insert_df.values)]
    return operation, records

def group_by_summary(df):
    # group_columns = ['factory_id', 'process_id', 'equipment_id', 'equipment_name', 'material', 'routing_revision', 'so', 'gop', 'qty']
    # g = df.groupby(group_columns)
    # insert_df = pd.DataFrame(list(g.groups.keys()), columns=group_columns)
    # columns, statistics = tuple(['total', 'spindle', 'idle']), tuple(['avg', 'max', 'min'])
    # df_ = df.rename(columns={'idle_inter':'idle'}).copy()
    # for c in columns:
    #     g = df_.groupby(['equipment_id', 'material', 'routing_revision', 'so', 'gop'])[f'{c}']
    #     groups = g.mean(), g.max(), g.min()
    #     for statistic, groups_ in zip(statistics, groups):
    #         g_ = groups_.reset_index(['equipment_id', 'material', 'routing_revision', 'so', 'gop']).rename(columns={f'{c}':f'{c}_cycle_{statistic}'})[['equipment_id', 'material', 'routing_revision', 'so', 'gop', f'{c}_cycle_{statistic}']]
    #         g_[f'{c}_cycle_{statistic}'] = np.round(g_[f'{c}_cycle_{statistic}'].values, 3)
    #         insert_df = insert_df.merge(g_, on=['equipment_id', 'material', 'routing_revision', 'so', 'gop'])
    # insert_df = insert_df.sort_values(['equipment_name']).reset_index(drop=True)

    group_columns = ['factory_id', 'process_id', 'equipment_id', 'equipment_name', 'material', 'routing_revision', 'so', 'gop', 'qty']
    g = df.groupby(group_columns)
    true_cycle = df.groupby(group_columns)['cycle_number'].max().reset_index().rename(columns={'cycle_number':'cycle'})
    insert_df = pd.DataFrame(list(g.groups.keys()), columns=group_columns)
    columns, statistics = tuple(['total', 'spindle', 'idle']), tuple(['avg', 'max', 'min'])
    df_ = df.rename(columns={'idle_inter':'idle'}).copy()
    for c in columns:
        g = df_.groupby(['equipment_id', 'material', 'routing_revision', 'so', 'gop', 'qty'])[f'{c}']
        groups = g.mean(), g.max(), g.min()
        for statistic, groups_ in zip(statistics, groups):
            g_ = groups_.reset_index(['equipment_id', 'material', 'routing_revision', 'so', 'gop', 'qty']).rename(columns={f'{c}':f'{c}_cycle_{statistic}'})[['equipment_id', 'material', 'routing_revision', 'so', 'gop', 'qty', f'{c}_cycle_{statistic}']]
            g_[f'{c}_cycle_{statistic}'] = np.round(g_[f'{c}_cycle_{statistic}'].values, 3)
            insert_df = insert_df.merge(g_, on=['equipment_id', 'material', 'routing_revision', 'so', 'gop', 'qty'])
    insert_df = insert_df.sort_values(['equipment_name']).reset_index(drop=True)
    insert_df = insert_df.merge(true_cycle, on=group_columns)

    return insert_df