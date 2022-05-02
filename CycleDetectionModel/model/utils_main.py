import pandas as pd
import numpy as np
import config
import json
import factoroid as fr

from model.utils_function import *
from model.utils_cycle import *
from model.utils_output import *

def main_w_so(so_row, plotting, eqm_, stop_times, idx, company):
    WebSQL = fr.connector.MySQLDB('db-01')
    DevSQL = fr.connector.MySQLDB('db-dev')

    order_number_, gop_, material_, quantity_ = so_row['so'], so_row['gop'], so_row['material'], int(so_row['qty'])
    total_runtime_, start_, stop_ = so_row['timediff'], str(so_row['start_time'] - pd.DateOffset(minutes=15)), str(so_row['stop_time'] + pd.DateOffset(minutes=0))
    # period_ = fr.utils.Period([start_, stop_])
    print(eqm_, order_number_, gop_, material_, quantity_, start_, stop_)

    machine, order_number = eqm_, order_number_

    ###########################################################################
    # read parameter
    ###########################################################################
    with open('equipment_properties.json', 'r') as fin:
        parameter = json.load(fin)

    gap = parameter[machine]['gap']
    threshold = parameter[machine]['threshold']
    min_threshold = parameter[machine]['min_threshold_value']
    smooth_factor = parameter[machine]['smooth_factor']
    step = parameter[machine]['step']
    upper_limit = parameter[machine]['upper']
    poly_order = parameter[machine]['order']
    mac_main = parameter[machine]['mac']
    door_sensors = parameter[machine]['door_sensor']
    window = parameter[machine]['window']
    max_cycle = parameter[machine]['max_cycle']
    idle_value = parameter[machine]['idle_value']
    config.plot = plotting
    config.company = company

    timeWindowRedefine = False
    while True:
        #  initialization
        patterns, candidates, pattern_index, start_time, _, lll, _ = pattern_history_parse(machine, order_number)
        print(patterns)
        if timeWindowRedefine == False:
            if start_time != "":
                start_time = start_time.replace("T", " ")[:19]
                # print(patterns, start_, start_time)
                start_ = max(start_, start_time)
        else:
            print("Current start time: ", start_)
            start_ = input("Proposed new start datetime (format: YYYY-MM-DD HH:MM:SS): ")

        # let's check next SO mapping if available
        if idx + 1 < len(stop_times):
            stop_time = str(stop_times[idx + 1]).replace("T", " ")[:19]
        else:
            stop_time = str(so_row['stop_time'] + pd.DateOffset(hours=total_runtime_))

        period_ = fr.utils.Period([start_, stop_time])
        print("Current time window of interest: ", start_, period_)

        row = WebSQL.get_data(mac=mac_main, period=period_.utc)
        #data = row[['timestamp', 'mac', 'major']].rename(columns={'major': 'value'}).copy()
        data = fr.processor.preprocess(row)
        if property == 100:
            data.loc[data['value'] > 11900, 'value'] = -10
        # data = data_[data_['timestamp'].diff().dt.total_seconds() >= 0.5].reset_index(drop=True) if not data_.empty else data_
        # timestamp data type conversion
        data2, data = data[data['timestamp'] > str(pd.to_datetime(stop_))], data[data['timestamp'] <= str(pd.to_datetime(stop_))]
        data2["timestamp"] = pd.to_datetime(data2["timestamp"])
        data["timestamp"] = pd.to_datetime(data["timestamp"])
        if plotting:
            plt.figure(figsize=(25, 2))
            plt.plot(data['timestamp'], data['value'], label='current SO')
            plt.plot(data2['timestamp'], data2['value'], alpha=0.7, label='next SO')
            plt.legend()
            plt.show()
        temp = data[data['mac'] == mac_main]

        # cleaning + smoothing + inactive phase detection
        times_, y_ = cleaning(temp, smooth_factor, min_threshold)
        smooth_ = smoothing(y_, step)
        times, ss, inactive_cycles_ = inactive_phase_cleaning_normalization(smooth_, times_, upper_limit, window)

        # active phase detection
        cycles_ = active_phase(ss, times, threshold, upper_limit, max_cycle * 60, gap, idle_value)
        cycles = cycle_generate(cycles_, times, ss, inactive_cycles_)

        # validate if we can see at least one complete cycle; otherwise, let's adjust the start time if needed
        proceed = input("Do we need to change time window to detect at least one complete cycle? (Y/N): ")
        if proceed == "N":
            break
        timeWindowRedefine = True


    # filter some of active phase with broken intervals
    cycles = [x for x in cycles if min([y[1] for y in x]) <= 0.3]

    # cycle parameter
    cycle_parameter = parameter[machine]['cycle']
    cycle_parameter['L'] = 500 if cycle_parameter['lower_limit'] <= 100 else 1000
    cycle_parameter['min_dist'] = round(max(cycle_parameter['lower_limit'] ** 0.4, 2.8), 1) * 1.3
    cycle_parameter['order'] = int(min(max(cycle_parameter['lower_limit'] ** 0.6, 5), 15))
    cycle_parameter['gap'] = 50
    cycle_parameter['n'] = 5
    cycle_parameter['cycle_gap'] = [30, 60][machine not in {"APEC G3020 #1"}]

    # find cycles! - utils_cycle.py
    patterns, candidates, first_phase, last_phase = pattern_detector(cycles, cycle_parameter, machine, order_number)
    output, complete_cycle = best_cycle(patterns, quantity_)
    #final = [0, 0, [xx for xx in sorted(complete_cycle) if xx[1] != -1]]

    # check/build cycles based on our findings with possible adjustmnet! - utils_output.py
    output, patterns, candidates, machine, complete_cycle, first_phase, last_phase, flag = so_pattern_output(output, patterns, candidates, machine, complete_cycle, times, ss, cycles, first_phase, last_phase,  material_, order_number, cycle_parameter)

    # if the flag is True (pattern was found manually, let's revise the parameter file
    if flag and output != []:
        timeInterval = [patterns[i][0]//60 for i in output]
        if max(timeInterval) > cycle_parameter['upper_limit']:
            cycle_parameter['upper_limit'] = int(max(timeInterval) * 1.1)
        if min(timeInterval) < cycle_parameter['lower_limit']:
            cycle_parameter['lower_limit'] = int(min(timeInterval) * 0.9)
        parameter[machine]['cycle'] = cycle_parameter
        print("Updating the parameter with the latest information")
        with open('equipment_properties.json', 'w') as fout:
            json.dump(parameter, fout)

    return output, patterns, candidates, machine, complete_cycle, times, ss, cycles, first_phase, last_phase, data


def main_wo_so(plotting, eqm_, now, company):
    WebSQL = fr.connector.MySQLDB('db-01')
    #DevSQL = fr.connector.MySQLDB('db-dev')

    #order_number_, gop_, material_, quantity_ = so_row['order_number'], so_row['gop'], so_row['material'], int(so_row['qty'])
    material_, order_number, quantity_ = "", "", -1
    total_runtime_, start_, stop_ = 24, str(now - pd.DateOffset(hours=24)), str(now - pd.DateOffset(hours=0))
    # period_ = fr.utils.Period([start_, stop_])
    #print(eqm_, order_number_, gop_, material_, quantity_)

    machine, order_number = eqm_, ""

    ###########################################################################
    # read parameter
    ###########################################################################
    with open('equipment_properties.json', 'r') as fin:
        parameter = json.load(fin)

    gap = parameter[machine]['gap']
    threshold = parameter[machine]['threshold']
    min_threshold = parameter[machine]['min_threshold_value']
    smooth_factor = parameter[machine]['smooth_factor']
    step = parameter[machine]['step']
    upper_limit = parameter[machine]['upper']
    poly_order = parameter[machine]['order']
    mac_main = parameter[machine]['mac']
    door_sensors = parameter[machine]['door_sensor']
    window = parameter[machine]['window']
    max_cycle = parameter[machine]['max_cycle']
    idle_value = parameter[machine]['idle_value']
    config.plot = plotting
    config.company = company

    timeWindowRedefine = False
    while True:
        #  initialization
        patterns, candidates, pattern_index, start_time, _, lll, _ = pattern_history_parse(machine, order_number)
        print(patterns)
        if timeWindowRedefine == False:
            if start_time != "":
                start_time = start_time.replace("T", " ")[:19]
                # print(patterns, start_, start_time)
                start_ = max(start_, start_time)
        else:
            print("Current start time: ", start_)
            start_ = input("Proposed new start datetime (format: YYYY-MM-DD HH:MM:SS): ")

        period_ = fr.utils.Period([start_, stop_])
        print("Current time window of interest: ", start_, period_)

        row = WebSQL.get_data(mac=mac_main, period=period_.utc)
        #data = row[['timestamp', 'mac', 'major']].rename(columns={'major': 'value'}).copy()
        data = fr.processor.preprocess(row)
        if property == 100:
            data.loc[data['value'] > 11900, 'value'] = -10
        # data = data_[data_['timestamp'].diff().dt.total_seconds() >= 0.5].reset_index(drop=True) if not data_.empty else data_
        # timestamp data type conversion
        data2, data = data[data['timestamp'] > str(pd.to_datetime(stop_))], data[data['timestamp'] <= str(pd.to_datetime(stop_))]
        data2["timestamp"] = pd.to_datetime(data2["timestamp"])
        data["timestamp"] = pd.to_datetime(data["timestamp"])
        if plotting:
            plt.figure(figsize=(24, 3))
            plt.plot(data['timestamp'], data['value'], label='current SO')
            plt.plot(data2['timestamp'], data2['value'], alpha=0.7, label='next SO')
            plt.legend()
            plt.show()
        temp = data[data['mac'] == mac_main]

        # cleaning + smoothing + inactive phase detection
        times_, y_ = cleaning(temp, smooth_factor, min_threshold)
        smooth_ = smoothing(y_, step)
        times, ss, inactive_cycles_ = inactive_phase_cleaning_normalization(smooth_, times_, upper_limit, window)

        # active phase detection
        cycles_ = active_phase(ss, times, threshold, upper_limit, max_cycle * 60, gap, idle_value)
        cycles = cycle_generate(cycles_, times, ss, inactive_cycles_)

        # validate if we can see at least one complete cycle; otherwise, let's adjust the start time if needed
        proceed = input("Do we need to change time window to detect at least one complete cycle? (Y/N): ")
        if proceed == "N":
            break
        timeWindowRedefine = True


    # filter some of active phase with broken intervals
    cycles = [x for x in cycles if min([y[1] for y in x]) <= 0.3]

    # cycle parameter
    cycle_parameter = parameter[machine]['cycle']
    mid = (cycle_parameter['lower_limit'] + cycle_parameter['upper_limit']) / 2
    cycle_parameter['L'] = 500 if cycle_parameter['lower_limit'] <= 100 else 1000
    cycle_parameter['min_dist'] = round(max(mid ** 0.4, 2.8), 1) * 1.5
    cycle_parameter['order'] = int(min(max(mid ** 0.6, 5), 15))
    if machine in {"L280-3"}:
        cycle_parameter['gap'] = 3
    elif machine in {"KREUZ CHAMFERING 머신", "7.5톤 브로치 M/C", "KN-152 호빙머신 1호기", "SE25A 기어셰이핑머신 2호", "전자빔용접기 EBM-6LB 1호기",
                     "플로포밍 2호기", "흡빙 M/C KA220CNC-2호기"}:
        cycle_parameter['gap'] = 1
    else:
        cycle_parameter['gap'] = 50
    cycle_parameter['n'] = 5
    cycle_parameter['cycle_gap'] = [30, 60][machine not in {"APEC G3020 #1"}]

    # find cycles! - utils_cycle.py
    patterns, candidates, first_phase, last_phase = pattern_detector(cycles, cycle_parameter, machine, order_number)
    output, complete_cycle = best_cycle(patterns, -1) # quantity is -1 if no SO mapping is available
    #final = [0, 0, [xx for xx in sorted(complete_cycle) if xx[1] != -1]]

    # check/build cycles based on our findings with possible adjustmnet! - utils_output.py
    output, patterns, candidates, machine, complete_cycle, first_phase, last_phase, flag = so_pattern_output(output, patterns, candidates, machine, complete_cycle, times, ss, cycles, first_phase, last_phase,  material_, order_number, cycle_parameter)

    # if the flag is True (pattern was found manually, let's revise the parameter file
    if flag and output != []:
        timeInterval = [patterns[i][0]//60 for i in output]
        if max(timeInterval) > cycle_parameter['upper_limit']:
            cycle_parameter['upper_limit'] = int(max(timeInterval) * 1.1)
        if min(timeInterval) < cycle_parameter['lower_limit']:
            cycle_parameter['lower_limit'] = int(min(timeInterval) * 0.9)
        parameter[machine]['cycle'] = cycle_parameter
        print("Updating the parameter with the latest information")
        with open('equipment_properties.json', 'w') as fout:
            json.dump(parameter, fout)

    return output, patterns, candidates, machine, complete_cycle, times, ss, cycles, first_phase, last_phase, data, mac_main


def main(machine, order_number, df_set, plotting=True):

    # make sure we have the right data type
    df_set["timestamp"] = pd.to_datetime(df_set["timestamp"])

    with open('equipment_properties.json', 'r') as fin:
        parameter = json.load(fin)

    gap = parameter[machine]['gap']
    threshold = parameter[machine]['threshold']
    min_threshold = parameter[machine]['min_threshold_value']
    smooth_factor = parameter[machine]['smooth_factor']
    step = parameter[machine]['step']
    upper_limit = parameter[machine]['upper']
    poly_order = parameter[machine]['order']
    mac_main = parameter[machine]['mac']
    door_sensors = parameter[machine]['door_sensor']
    window = parameter[machine]['window']
    max_cycle = parameter[machine]['max_cycle']
    idle_value = parameter[machine]['idle_value']
    config.plot = plotting

    temp = df_set[df_set['mac'] == mac_main]

    #  initialization
    patterns, candidates, pattern_index, start_time, _, lll, _ = pattern_history_parse(machine, order_number)
    #print(patterns)

    # cleaning + smoothing + inactive phase detection
    times_, y_ = cleaning(temp, smooth_factor, min_threshold)
    smooth_ = smoothing(y_, step)
    times, ss, inactive_cycles_ = inactive_phase_cleaning_normalization(smooth_, times_, upper_limit, window)

    # active phase detection
    cycles_ = active_phase(ss, times, threshold, upper_limit, max_cycle * 60, gap, idle_value)
    cycles = cycle_generate(cycles_, times, ss, inactive_cycles_)

    # cycle parameter
    cycle_parameter = parameter[machine]['cycle']
    cycle_parameter['L'] = 500 if cycle_parameter['lower_limit'] <= 100 else 1000
    cycle_parameter['min_dist'] = round(max(cycle_parameter['lower_limit'] ** 0.4, 2.5), 1)
    cycle_parameter['order'] = int(min(max(cycle_parameter['lower_limit'] ** 0.6, 5), 15))
    cycle_parameter['gap'] = 50
    cycle_parameter['n'] = 5

    # find cycles!
    patterns, candidates, first_phase, last_phase = pattern_detector(cycles, cycle_parameter, machine, order_number)
    output, complete_cycle = best_cycle(patterns)
    final = [0, 0, [xx for xx in sorted(complete_cycle) if xx[1] != -1]]
    store_cycle_output_and_plot(output, patterns, candidates, machine, complete_cycle, times, ss, cycles, first_phase,
                                last_phase, order_number)


    return output, patterns, candidates, machine, complete_cycle, times, ss, cycles, first_phase, last_phase
