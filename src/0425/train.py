import pandas as pd
import numpy as np
from datetime import datetime
from datetime import timedelta

def load_train_data():
    df = pd.read_csv('../../dataSets/training/training_20min_avg_travel_time.csv')
    trajectories_train = {}
    start_time  = datetime.strptime("2016-07-19 00:00:00",'%Y-%m-%d %H:%M:%S')

    for idx in range(df.shape[0]):
        line = df.iloc[idx]
        route = str(line['intersection_id']+str(line['tollgate_id']))
        stime = datetime.strptime(line['time_window'][1:20],'%Y-%m-%d %H:%M:%S')
        day = (stime - start_time).days
        tw_n = (stime.hour * 60 + stime.minute)/20

        if route not in trajectories_train.keys():
            trajectories_train[route] = np.zeros(72*91).reshape(91,72)
        trajectories_train[route][day][tw_n] = line['avg_travel_time']
    return trajectories_train

#input:route路twn时间窗time时间
#output:time往前星期相同的平均值~~
def get_history_mean(data,route,twn,time):
    #time=datetime.strptime(date,'%Y-%m-%d %H:%M:%S')
    start_time  = datetime.strptime("2016-07-19 00:00:00",'%Y-%m-%d %H:%M:%S')
    startday=(time-start_time).days
    #print startday
    ##筛选出同期的历史数据
    history_data=[]
    while(startday>-1):
        if(startday<90):
            cur_data=data[route][startday][twn]
            history_data.append(cur_data)
        startday-=7

    max_value = np.max(history_data)
    min_value = np.min(history_data)
    final_x = []
    for i in range(len(history_data)):
        if history_data[i]>min_value and history_data[i]<max_value:
            final_x.append(history_data[i])
    final_x = np.array(final_x)
    if np.sum(final_x) == 0:
        return 0
    else:
        return np.average(final_x,weights=range(1,final_x.shape[0]+1))
    #return np.average(final_x)

def predict_mean():
    trajectories_train = load_train_data()
    pstart_time=datetime.strptime("2016-10-18 00:00:00",'%Y-%m-%d %H:%M:%S')
    y_tw_list = [24,25,26,27,28,29,51,52,53,54,55,56]
    #y_tw_list = [18,19,20,21,22,23,45,46,47,48,49,50]
    test_data = {}
    for route in trajectories_train.keys():
        for twn in y_tw_list:
            for day in range(7):
                if route not in test_data.keys():
                    test_data[route] = np.zeros(7*72).reshape(7,72)
                test_data[route][day][twn] = get_history_mean(trajectories_train,route,twn,pstart_time+timedelta(days=day))
    return test_data

def LASSO(data,route,twn,time):
    start_time  = datetime.strptime("2016-07-19 00:00:00",'%Y-%m-%d %H:%M:%S')
    startday=(time-start_time).days
    #print startday
    ##筛选出同期的历史数据
    history_data=[]
    while(startday>-1):
        if(startday<90):
            cur_data=data[route][startday][twn-6:twn+1]
            history_data.append(cur_data)
        startday-=7
    max_value = np.max(history_data)
    min_value = np.min(history_data)
    final_x = [] ##用来训练lasso的数据[x,y]
    for i in range(len(history_data)):
        if history_data[i][6]>min_value and history_data[i][6]<max_value:
            if 0 not in history_data[i]:
                final_x.append(history_data[i])

    final_x = np.array(final_x)
    #print route,twn,time,final_x.shape
    from sklearn import linear_model
    clf = linear_model.Lasso(alpha=0.1)
    clf.fit(final_x[:,0:6],final_x[:,6])
    return clf

def get_lasso(data_origin,test_data,route,twn,day,time):
    clf = LASSO(data_origin,route,twn,time) ##用原始数据学模型
    return clf.predict(test_data[route][day][twn-6:twn]) ##用整合数据做预测


##加载测试数据前6窗数据
def load_test_data():
    df_test=pd.read_csv('../../dataSets/testing_phase1/trajectories_20min_avg_travel_time.csv')
    trajectories_test = {}
    start_time  = datetime.strptime("2016-10-18 00:00:00",'%Y-%m-%d %H:%M:%S')

    for idx in range(df_test.shape[0]):
        line = df_test.iloc[idx]
        route = str(line['intersection_id']+str(line['tollgate_id']))
        stime = datetime.strptime(line['time_window'][1:20],'%Y-%m-%d %H:%M:%S')
        day = (stime - start_time).days
        tw_n = (stime.hour * 60 + stime.minute)/20

        if route not in trajectories_test.keys():
            trajectories_test[route] = np.zeros(72*7).reshape(7,72)
        trajectories_test[route][day][tw_n] = line['avg_travel_time']
    return trajectories_test

def predict_lasso():
    '''
    加载数据，前6个窗来自于加载的测试数据，后六个窗来自于均值数据
    输出：Lasso预测的结果
    '''
    data1 = load_test_data()
    data2 = predict_mean()
    data_origin = load_train_data()
    data = {}
    for route in data1.keys():
        data[route] = data1[route]+data2[route] ##整合后的数据供Lasso预测使
    #print data['A2']
    pstart_time=datetime.strptime("2016-10-18 00:00:00",'%Y-%m-%d %H:%M:%S')
    y_tw_list = [24,25,26,27,28,29,51,52,53,54,55,56]
    #y_tw_list = [18,19,20,21,22,23,45,46,47,48,49,50]
    pred_data = {}
    for route in data.keys():
        for twn in y_tw_list:
            for day in range(7):
                if route not in pred_data.keys():
                    pred_data[route] = np.zeros(7*72).reshape(7,72)
                pred_data[route][day][twn] = get_lasso(data_origin,data,route,twn,day,pstart_time+timedelta(days=day))
    return pred_data


def out_data(test_data):
    out_data = test_data;
    ## 数据输出
    from datetime import datetime
    from datetime import timedelta
    from pandas import DataFrame


    intersection = []
    tollgate = []
    time_window = []
    avg_time = []
    routes = ['A2','A3',"B1","B3","C1","C3"]
    for route in routes:
        n_day = 7
        start_time = datetime.strptime('2016-10-18 08:00:00',"%Y-%m-%d %H:%M:%S")
        for day in range(n_day):
            starttime = start_time + timedelta(days=day)
            i = 0
            for k in range(24,30): #(51,57)（24，30）
                time_window.append('\"[' + (starttime + timedelta(seconds=1200*i)).strftime("%Y-%m-%d %H:%M:%S")\
                                   + "," + (starttime + timedelta(seconds=1200*i+1200)).strftime("%Y-%m-%d %H:%M:%S") + ')\"')
                intersection.append(str(route[0]))
                tollgate.append(str(route[1]))
                avg_time.append(out_data[route][day][k])
                i+=1

    for route in routes:
        n_day = 7
        start_time = datetime.strptime('2016-10-18 17:00:00',"%Y-%m-%d %H:%M:%S")
        for day in range(n_day):
            starttime = start_time + timedelta(days=day)
            i = 0
            for k in range(51,57): #(51,57)（24，30）
                time_window.append('\"[' + (starttime + timedelta(seconds=1200*i)).strftime("%Y-%m-%d %H:%M:%S")\
                                   + "," + (starttime + timedelta(seconds=1200*i+1200)).strftime("%Y-%m-%d %H:%M:%S") + ')\"')
                intersection.append(str(route[0]))
                tollgate.append(str(route[1]))
                avg_time.append(out_data[route][day][k])
                i+=1

    d = {"intersection_id":intersection,"tollgate_id":tollgate,"time_window":time_window,"avg_travel_time":avg_time}
    pd = DataFrame(data=d)
    pd.to_csv('out.csv',index=False,columns=["intersection_id","tollgate_id","time_window","avg_travel_time"])




## 学习
pred_mean_data = predict_mean()
pred_lasso_data = predict_lasso()
for route in pred_mean_data.keys():
    pred_data[route] = (pred_mean_data[route] + pred_lasso_data[route])/2.0 ##两种方法结果取均值

