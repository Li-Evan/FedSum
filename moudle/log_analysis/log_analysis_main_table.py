import numpy as np


def get_Centralized():

    str_1 = '''>> ROUGE-F(1/2/l): 37.40/15.46/29.91		>> ROUGE-F(1/2/l): 35.99/14.30/28.35		>> ROUGE-F(1/2/l): 29.38/8.85/22.51
>> ROUGE-R(1/2/l): 46.62/20.31/37.39		>> ROUGE-R(1/2/l): 44.16/18.41/34.91		>> ROUGE-R(1/2/l): 35.71/11.24/27.47
'''
    str_2 = '''>> ROUGE-F(1/2/l): 37.62/15.60/30.13		>> ROUGE-F(1/2/l): 29.40/8.68/22.48		>> ROUGE-F(1/2/l): 29.70/9.13/22.84
>> ROUGE-R(1/2/l): 47.25/20.67/37.93		>> ROUGE-R(1/2/l): 35.28/10.83/27.05		>> ROUGE-R(1/2/l): 36.19/11.66/27.94
'''
    str_3 = '''>> ROUGE-F(1/2/l): 26.75/7.31/20.27		>> ROUGE-F(1/2/l): 36.42/14.71/28.72		>> ROUGE-F(1/2/l): 29.13/8.61/22.27
>> ROUGE-R(1/2/l): 31.35/8.80/23.81		>> ROUGE-R(1/2/l): 44.68/18.91/35.36		>> ROUGE-R(1/2/l): 35.32/10.91/27.12
'''
    str_list = [str_1, str_2, str_3]

    epoch_commu_list = ["(2,5)", "(4,5)", "(6,5)"]
    epoch_commu_ROUGE_F_dict = {
        '(2,5)': [[], [], []],
        '(4,5)': [[], [], []],
        '(6,5)': [[], [], []],
    }

    epoch_commu_ROUGE_R_dict = {
        '(2,5)': [[], [], []],
        '(4,5)': [[], [], []],
        '(6,5)': [[], [], []],
    }


    for my_str in str_list:
        R_F_str, R_R_str = my_str.strip().split('\n')
        for i in range(len(epoch_commu_list)):
            split_strategy = epoch_commu_list[i]

            tmp_ROUGE_F_list = R_F_str.strip().split('\t\t')[i][18:].split('/')
            R_F_1, R_F_2, R_F_l = tmp_ROUGE_F_list
            R_F_1, R_F_2, R_F_l = float(R_F_1), float(R_F_2), float(R_F_l)
            epoch_commu_ROUGE_F_dict[split_strategy][0].append(R_F_1)
            epoch_commu_ROUGE_F_dict[split_strategy][1].append(R_F_2)
            epoch_commu_ROUGE_F_dict[split_strategy][2].append(R_F_l)

            tmp_ROUGE_R_list = R_R_str.strip().split('\t\t')[i][18:].split('/')
            R_R_1, R_R_2, R_R_l = tmp_ROUGE_R_list
            R_R_1, R_R_2, R_R_l = float(R_R_1), float(R_R_2), float(R_R_l)
            epoch_commu_ROUGE_R_dict[split_strategy][0].append(R_R_1)
            epoch_commu_ROUGE_R_dict[split_strategy][1].append(R_R_2)
            epoch_commu_ROUGE_R_dict[split_strategy][2].append(R_R_l)


    R_F_list = [[] for _ in range(len(epoch_commu_list))]
    R_R_list = [[] for _ in range(len(epoch_commu_list))]

    for i in range(len(epoch_commu_list)):
        split_strategy = epoch_commu_list[i]
        R1_avg = round(np.mean(epoch_commu_ROUGE_F_dict[split_strategy][0]), 2)
        R1_var = round(np.var(epoch_commu_ROUGE_F_dict[split_strategy][0]), 2)

        R2_avg = round(np.mean(epoch_commu_ROUGE_F_dict[split_strategy][1]), 2)
        R2_var = round(np.var(epoch_commu_ROUGE_F_dict[split_strategy][1]), 2)

        Rl_avg = round(np.mean(epoch_commu_ROUGE_F_dict[split_strategy][2]), 2)
        Rl_var = round(np.var(epoch_commu_ROUGE_F_dict[split_strategy][2]), 2)

        # R_F_list[i] = str(R1_avg) + "±" + str(R1_var) + "/" + str(R2_avg) + "±" + str(R2_var) + "/" + str(
        #     Rl_avg) + "±" + str(Rl_var)

        R_F_list[i] = str(R1_avg) + "/" + str(R2_avg) + "/" + str(Rl_avg) + "\t" + str(R1_var) + "/" + str(R2_var) + "/" + str(Rl_var)

    for i in range(len(epoch_commu_list)):
        split_strategy = epoch_commu_list[i]
        R1_avg = round(np.mean(epoch_commu_ROUGE_R_dict[split_strategy][0]), 2)
        R1_var = round(np.var(epoch_commu_ROUGE_R_dict[split_strategy][0]), 2)

        R2_avg = round(np.mean(epoch_commu_ROUGE_R_dict[split_strategy][1]), 2)
        R2_var = round(np.var(epoch_commu_ROUGE_R_dict[split_strategy][1]), 2)

        Rl_avg = round(np.mean(epoch_commu_ROUGE_R_dict[split_strategy][2]), 2)
        Rl_var = round(np.var(epoch_commu_ROUGE_R_dict[split_strategy][2]), 2)

        # R_R_list[i] = str(R1_avg) + "±" + str(R1_var) + "/" + str(R2_avg) + "±" + str(R2_var) + "/" + str(
        #     Rl_avg) + "±" + str(Rl_var)
        #
        R_R_list[i] = str(R1_avg) + "/" + str(R2_avg) + "/" + str(Rl_avg) + "\t" + str(R1_var) + "/" + str(R2_var) + "/" + str(Rl_var)

    print(('\t\t').join(R_F_list).strip('\''))
    print(('\t\t').join(R_R_list).strip('\''))

def record_pre_process(str_list):
    split_strategy_list = ["Dirichlet01", "Dirichlet05", "Dirichlet1", "Dirichlet8", "Uniform"]
    split_strategy_ROUGE_F_dict = {
        'Dirichlet01': [[], [], []],
        'Dirichlet05': [[], [], []],
        'Dirichlet1': [[], [], []],
        'Dirichlet8': [[], [], []],
        'Uniform': [[], [], []]
    }

    split_strategy_ROUGE_R_dict = {
        'Dirichlet01': [[], [], []],
        'Dirichlet05': [[], [], []],
        'Dirichlet1': [[], [], []],
        'Dirichlet8': [[], [], []],
        'Uniform': [[], [], []]
    }

    for my_str in str_list:
        R_F_str, R_R_str = my_str.strip().split('\n')
        for i in range(len(split_strategy_list)):
            split_strategy = split_strategy_list[i]

            tmp_ROUGE_F_list = R_F_str.strip().split('\t\t')[i][18:].split('/')
            R_F_1, R_F_2, R_F_l = tmp_ROUGE_F_list
            R_F_1, R_F_2, R_F_l = float(R_F_1), float(R_F_2), float(R_F_l)
            split_strategy_ROUGE_F_dict[split_strategy][0].append(R_F_1)
            split_strategy_ROUGE_F_dict[split_strategy][1].append(R_F_2)
            split_strategy_ROUGE_F_dict[split_strategy][2].append(R_F_l)

            tmp_ROUGE_R_list = R_R_str.strip().split('\t\t')[i][18:].split('/')
            R_R_1, R_R_2, R_R_l = tmp_ROUGE_R_list
            R_R_1, R_R_2, R_R_l = float(R_R_1), float(R_R_2), float(R_R_l)
            split_strategy_ROUGE_R_dict[split_strategy][0].append(R_R_1)
            split_strategy_ROUGE_R_dict[split_strategy][1].append(R_R_2)
            split_strategy_ROUGE_R_dict[split_strategy][2].append(R_R_l)

    # R_F_avg_list = [[] for _ in range(len(split_strategy_list))]
    # R_R_avg_list = [[] for _ in range(len(split_strategy_list))]
    #
    # R_F_var_list = [[] for _ in range(len(split_strategy_list))]
    # R_R_var_list = [[] for _ in range(len(split_strategy_list))]

    R_F_list = [[] for _ in range(len(split_strategy_list))]
    R_R_list = [[] for _ in range(len(split_strategy_list))]

    for i in range(len(split_strategy_list)):
        split_strategy = split_strategy_list[i]
        R1_avg = round(np.mean(split_strategy_ROUGE_F_dict[split_strategy][0]), 2)
        R1_var = round(np.var(split_strategy_ROUGE_F_dict[split_strategy][0]), 2)

        R2_avg = round(np.mean(split_strategy_ROUGE_F_dict[split_strategy][1]), 2)
        R2_var = round(np.var(split_strategy_ROUGE_F_dict[split_strategy][1]), 2)

        Rl_avg = round(np.mean(split_strategy_ROUGE_F_dict[split_strategy][2]), 2)
        Rl_var = round(np.var(split_strategy_ROUGE_F_dict[split_strategy][2]), 2)

        # R_F_list[i] = str(R1_avg) + "±" + str(R1_var) + "/" + str(R2_avg) + "±" + str(R2_var) + "/" + str(
        #     Rl_avg) + "±" + str(Rl_var)

        R_F_list[i] = str(R1_avg) + "/" + str(R2_avg) + "/" + str(Rl_avg) + "\t" + str(R1_var) + "/" + str(
            R2_var) + "/" + str(Rl_var)

    for i in range(len(split_strategy_list)):
        split_strategy = split_strategy_list[i]
        R1_avg = round(np.mean(split_strategy_ROUGE_R_dict[split_strategy][0]), 2)
        R1_var = round(np.var(split_strategy_ROUGE_R_dict[split_strategy][0]), 2)

        R2_avg = round(np.mean(split_strategy_ROUGE_R_dict[split_strategy][1]), 2)
        R2_var = round(np.var(split_strategy_ROUGE_R_dict[split_strategy][1]), 2)

        Rl_avg = round(np.mean(split_strategy_ROUGE_R_dict[split_strategy][2]), 2)
        Rl_var = round(np.var(split_strategy_ROUGE_R_dict[split_strategy][2]), 2)

        # R_R_list[i] = str(R1_avg) + "±" + str(R1_var) + "/" + str(R2_avg) + "±" + str(R2_var) + "/" + str(
        #     Rl_avg) + "±" + str(Rl_var)

        R_R_list[i] = str(R1_avg) + "/" + str(R2_avg) + "/" + str(Rl_avg) + "\t" + str(R1_var) + "/" + str(
            R2_var) + "/" + str(Rl_var)

    print(('\t\t').join(R_F_list).strip('\''))
    print(('\t\t').join(R_R_list).strip('\''))


if __name__ == '__main__':
    print("Centralized")
    get_Centralized()
    ############################################################

    Separate = [
        '''>> ROUGE-F(1/2/l): 32.10/11.40/25.21		>> ROUGE-F(1/2/l): 34.06/12.99/27.26		>> ROUGE-F(1/2/l): 30.66/10.22/23.93		>> ROUGE-F(1/2/l): 30.91/10.06/23.83		>> ROUGE-F(1/2/l): 31.30/10.66/24.49
    >> ROUGE-R(1/2/l): 39.49/14.63/31.08		>> ROUGE-R(1/2/l): 42.69/17.02/34.22		>> ROUGE-R(1/2/l): 37.49/13.00/29.33		>> ROUGE-R(1/2/l): 37.87/12.90/29.29		>> ROUGE-R(1/2/l): 38.46/13.68/30.18
    ''',
        '''>> ROUGE-F(1/2/l): 31.20/10.68/24.42		>> ROUGE-F(1/2/l): 31.93/10.94/24.82		>> ROUGE-F(1/2/l): 31.75/11.03/24.89		>> ROUGE-F(1/2/l): 28.03/8.12/21.44		>> ROUGE-F(1/2/l): 32.22/11.39/25.32
    >> ROUGE-R(1/2/l): 38.14/13.59/29.92		>> ROUGE-R(1/2/l): 39.26/14.05/30.61		>> ROUGE-R(1/2/l): 39.20/14.23/30.80		>> ROUGE-R(1/2/l): 33.53/10.10/25.73		>> ROUGE-R(1/2/l): 39.57/14.57/31.14
    ''',
        '''>> ROUGE-F(1/2/l): 33.34/12.47/26.52		>> ROUGE-F(1/2/l): 34.16/13.06/27.32		>> ROUGE-F(1/2/l): 33.24/12.38/26.40		>> ROUGE-F(1/2/l): 33.23/12.34/26.29		>> ROUGE-F(1/2/l): 31.25/10.57/24.36
    >> ROUGE-R(1/2/l): 41.46/16.17/33.04		>> ROUGE-R(1/2/l): 42.74/17.08/34.24		>> ROUGE-R(1/2/l): 41.32/16.06/32.88		>> ROUGE-R(1/2/l): 41.19/15.97/32.68		>> ROUGE-R(1/2/l): 38.27/13.50/29.92
    '''
    ]

    print("Separate")
    record_pre_process(Separate)
    ############################################################
    FedAvg = [
        '''>> ROUGE-F(1/2/l): 32.89/12.06/26.05		>> ROUGE-F(1/2/l): 35.26/13.92/28.21		>> ROUGE-F(1/2/l): 33.79/12.82/26.93		>> ROUGE-F(1/2/l): 33.83/12.86/27.01		>> ROUGE-F(1/2/l): 31.89/11.11/24.99
>> ROUGE-R(1/2/l): 40.85/15.61/32.41		>> ROUGE-R(1/2/l): 43.79/18.08/35.10		>> ROUGE-R(1/2/l): 42.16/16.70/33.66		>> ROUGE-R(1/2/l): 42.29/16.81/33.82		>> ROUGE-R(1/2/l): 39.33/14.33/30.90
''',
        '''>> ROUGE-F(1/2/l): 34.52/12.97/27.37		>> ROUGE-F(1/2/l): 34.03/13.00/27.27		>> ROUGE-F(1/2/l): 33.75/12.53/26.70		>> ROUGE-F(1/2/l): 34.26/13.18/27.43		>> ROUGE-F(1/2/l): 33.13/12.27/26.31
>> ROUGE-R(1/2/l): 43.40/17.16/34.49		>> ROUGE-R(1/2/l): 42.61/16.99/34.19		>> ROUGE-R(1/2/l): 42.15/16.43/33.42		>> ROUGE-R(1/2/l): 42.90/17.25/34.41		>> ROUGE-R(1/2/l): 41.12/15.89/32.71
''',
        '''>> ROUGE-F(1/2/l): 34.92/13.58/28.08		>> ROUGE-F(1/2/l): 29.94/9.73/23.10		>> ROUGE-F(1/2/l): 32.64/11.81/25.82		>> ROUGE-F(1/2/l): 34.15/13.09/27.33		>> ROUGE-F(1/2/l): 34.87/13.57/28.03
>> ROUGE-R(1/2/l): 44.16/17.99/35.54		>> ROUGE-R(1/2/l): 36.03/12.19/27.89		>> ROUGE-R(1/2/l): 40.54/15.32/32.14		>> ROUGE-R(1/2/l): 42.79/17.14/34.29		>> ROUGE-R(1/2/l): 43.94/17.92/35.36
'''
    ]

    print("FedAvg")
    record_pre_process(FedAvg)
    ############################################################
    FedDC = [
        '''>> ROUGE-F(1/2/l): 29.48/8.87/22.62		>> ROUGE-F(1/2/l): 33.81/12.77/26.89		>> ROUGE-F(1/2/l): 34.02/12.97/27.14		>> ROUGE-F(1/2/l): 35.59/14.05/28.60		>> ROUGE-F(1/2/l): 31.34/10.58/24.34
>> ROUGE-R(1/2/l): 35.73/11.21/27.51		>> ROUGE-R(1/2/l): 42.15/16.63/33.58		>> ROUGE-R(1/2/l): 42.49/16.93/33.97		>> ROUGE-R(1/2/l): 44.85/18.60/36.10		>> ROUGE-R(1/2/l): 37.98/13.35/29.58
''',
        '''>> ROUGE-F(1/2/l): 34.48/13.15/27.36		>> ROUGE-F(1/2/l): 29.89/9.28/23.04		>> ROUGE-F(1/2/l): 33.94/12.71/26.95		>> ROUGE-F(1/2/l): 36.08/14.47/29.16		>> ROUGE-F(1/2/l): 32.91/12.09/26.03
>> ROUGE-R(1/2/l): 43.12/17.25/34.28		>> ROUGE-R(1/2/l): 36.52/11.90/28.26		>> ROUGE-R(1/2/l): 42.49/16.67/33.81		>> ROUGE-R(1/2/l): 45.75/19.31/37.01		>> ROUGE-R(1/2/l): 40.81/15.63/32.34
''',
        '''>> ROUGE-F(1/2/l): 34.76/13.59/27.95		>> ROUGE-F(1/2/l): 29.46/8.91/22.61		>> ROUGE-F(1/2/l): 34.44/13.25/27.43		>> ROUGE-F(1/2/l): 36.36/14.61/29.39		>> ROUGE-F(1/2/l): 35.05/13.76/28.25
>> ROUGE-R(1/2/l): 43.60/17.83/35.09		>> ROUGE-R(1/2/l): 35.84/11.36/27.62		>> ROUGE-R(1/2/l): 42.84/17.23/34.18		>> ROUGE-R(1/2/l): 46.29/19.59/37.46		>> ROUGE-R(1/2/l): 44.16/18.15/35.63
'''
    ]

    print("FedDC")
    record_pre_process(FedDC)
    ############################################################
    FedProx = [
        '''>> ROUGE-F(1/2/l): 34.54/13.19/27.61		>> ROUGE-F(1/2/l): 29.33/8.76/22.46		>> ROUGE-F(1/2/l): 31.69/10.79/24.72		>> ROUGE-F(1/2/l): 33.77/12.78/26.91		>> ROUGE-F(1/2/l): 33.42/12.43/26.36
>> ROUGE-R(1/2/l): 43.48/17.41/34.81		>> ROUGE-R(1/2/l): 35.69/11.16/27.44		>> ROUGE-R(1/2/l): 39.02/13.89/30.52		>> ROUGE-R(1/2/l): 42.15/16.65/33.65		>> ROUGE-R(1/2/l): 41.30/16.02/32.67
''',
        '''>> ROUGE-F(1/2/l): 35.62/14.15/28.81		>> ROUGE-F(1/2/l): 30.00/9.08/23.03		>> ROUGE-F(1/2/l): 34.54/13.25/27.65		>> ROUGE-F(1/2/l): 34.35/13.28/27.61		>> ROUGE-F(1/2/l): 28.04/8.24/21.47
>> ROUGE-R(1/2/l): 45.04/18.77/36.46		>> ROUGE-R(1/2/l): 36.14/11.40/27.83		>> ROUGE-R(1/2/l): 43.48/17.49/34.86		>> ROUGE-R(1/2/l): 43.04/17.40/34.65		>> ROUGE-R(1/2/l): 33.38/10.16/25.63
''',
        '''>> ROUGE-F(1/2/l): 35.79/14.22/28.91		>> ROUGE-F(1/2/l): 34.47/13.26/27.54		>> ROUGE-F(1/2/l): 35.07/13.64/28.17		>> ROUGE-F(1/2/l): 34.62/13.10/27.51		>> ROUGE-F(1/2/l): 30.66/10.22/23.87
>> ROUGE-R(1/2/l): 45.41/18.95/36.70		>> ROUGE-R(1/2/l): 43.26/17.43/34.62		>> ROUGE-R(1/2/l): 44.40/18.13/35.71		>> ROUGE-R(1/2/l): 43.66/17.34/34.76		>> ROUGE-R(1/2/l): 37.10/12.83/28.96
'''
    ]

    print("FedProx")
    record_pre_process(FedProx)
    ############################################################
    Scaffold = [
        '''>> ROUGE-F(1/2/l): 29.83/9.58/23.19		>> ROUGE-F(1/2/l): 31.96/11.20/25.09		>> ROUGE-F(1/2/l): 33.03/12.23/26.26		>> ROUGE-F(1/2/l): 34.01/12.96/27.24		>> ROUGE-F(1/2/l): 32.79/12.02/25.99
>> ROUGE-R(1/2/l): 36.21/12.08/28.21		>> ROUGE-R(1/2/l): 39.47/14.42/31.06		>> ROUGE-R(1/2/l): 41.05/15.85/32.70		>> ROUGE-R(1/2/l): 42.65/17.00/34.20		>> ROUGE-R(1/2/l): 40.65/15.54/32.28
''',
        '''>> ROUGE-F(1/2/l): 32.52/11.19/25.33		>> ROUGE-F(1/2/l): 31.24/10.81/24.40		>> ROUGE-F(1/2/l): 34.00/12.84/27.13		>> ROUGE-F(1/2/l): 33.76/12.81/26.97		>> ROUGE-F(1/2/l): 31.63/11.02/24.81
>> ROUGE-R(1/2/l): 39.90/14.35/31.16		>> ROUGE-R(1/2/l): 37.85/13.62/29.65		>> ROUGE-R(1/2/l): 42.64/16.85/34.07		>> ROUGE-R(1/2/l): 42.20/16.71/33.76		>> ROUGE-R(1/2/l): 38.81/14.08/30.50
''',
        '''>> ROUGE-F(1/2/l): 30.54/10.22/23.87		>> ROUGE-F(1/2/l): 33.66/12.69/26.80		>> ROUGE-F(1/2/l): 29.03/9.03/22.37		>> ROUGE-F(1/2/l): 32.96/12.18/26.17		>> ROUGE-F(1/2/l): 34.49/13.13/27.51
>> ROUGE-R(1/2/l): 37.18/12.96/29.13		>> ROUGE-R(1/2/l): 41.97/16.52/33.49		>> ROUGE-R(1/2/l): 34.73/11.21/26.85		>> ROUGE-R(1/2/l): 40.94/15.78/32.57		>> ROUGE-R(1/2/l): 43.42/17.33/34.70
''']

    print("Scaffold")
    record_pre_process(Scaffold)
    ############################################################
    FedNova = [
        '''>> ROUGE-F(1/2/l): 34.17/13.08/27.15		>> ROUGE-F(1/2/l): 32.19/10.99/25.05		>> ROUGE-F(1/2/l): 32.96/12.14/26.08		>> ROUGE-F(1/2/l): 33.44/12.49/26.52		>> ROUGE-F(1/2/l): 33.31/12.39/26.40
>> ROUGE-R(1/2/l): 42.13/16.85/33.56		>> ROUGE-R(1/2/l): 39.32/14.04/30.70		>> ROUGE-R(1/2/l): 40.68/15.64/32.27		>> ROUGE-R(1/2/l): 41.35/16.14/32.87		>> ROUGE-R(1/2/l): 41.13/15.99/32.68
''',
        '''>> ROUGE-F(1/2/l): 32.82/12.01/25.98		>> ROUGE-F(1/2/l): 32.95/11.64/25.75		>> ROUGE-F(1/2/l): 32.73/11.89/25.82		>> ROUGE-F(1/2/l): 33.37/12.44/26.49		>> ROUGE-F(1/2/l): 33.37/12.43/26.47
>> ROUGE-R(1/2/l): 40.50/15.45/32.12		>> ROUGE-R(1/2/l): 40.46/14.99/31.71		>> ROUGE-R(1/2/l): 40.28/15.27/31.85		>> ROUGE-R(1/2/l): 41.30/16.09/32.85		>> ROUGE-R(1/2/l): 41.24/16.05/32.78
''',
        '''>> ROUGE-F(1/2/l): 33.19/12.32/26.30		>> ROUGE-F(1/2/l): 29.45/8.70/22.51		>> ROUGE-F(1/2/l): 33.03/12.17/26.12		>> ROUGE-F(1/2/l): 33.62/12.65/26.70		>> ROUGE-F(1/2/l): 33.39/12.44/26.47
>> ROUGE-R(1/2/l): 41.07/15.91/32.62		>> ROUGE-R(1/2/l): 35.66/10.99/27.35		>> ROUGE-R(1/2/l): 40.74/15.67/32.28		>> ROUGE-R(1/2/l): 41.57/16.34/33.08		>> ROUGE-R(1/2/l): 41.24/16.06/32.77
''']

    print("FedNova")
    record_pre_process(FedNova)



    
    print("\n Finish ")
