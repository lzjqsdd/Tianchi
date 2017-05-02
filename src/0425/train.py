# -*- coding:utf8-*-
import pandas as pd
import numpy as np
from datetime import datetime
from datetime import timedelta

def load_train_data():
    df = pd.read_csv('../../dataSets/training/0428/training_20min_avg_travel_time.csv')
    trajectories_train = {}
    start_time  = datetime.strptime("2016-07-19 00:00:00",'%Y-%m-%d %H:%M:%S')

    for idx in range(df.shape[0]):
        line = df.iloc[idx]
        route = str(line['intersection_id']+str(line['tollgate_id']))
        stime = datetime.strptime(line['time_window'][1:20],'%Y-%m-%d %H:%M:%S')
        day = (stime - start_time).days
        tw_n = (stime.hour * 60 + stime.minute)/20

        if route not in trajectories_train.keys():
            trajectories_train[route] = np.zeros(72*84).reshape(84,72)
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
        if(startday<84):
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
    pstart_time=datetime.strptime("2016-10-11 00:00:00",'%Y-%m-%d %H:%M:%S')
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
        if(startday<84):
            cur_data=data[route][startday][twn-6:twn+1]
            history_data.append(cur_data)
        startday-=7

    history_data = np.array(history_data)
    max_value = np.max(history_data[:,-1])
    min_value = np.min(history_data[:,-1])
    final_x = [] ##用来训练lasso的数据[x,y]
    for i in range(len(history_data)):
        if history_data[i][6]>min_value and history_data[i][6]<max_value:
            if 0 not in history_data[i]:
                final_x.append(history_data[i])
    final_x = np.array(final_x)
    #print route,twn,time,final_x.shape
    from sklearn import linear_model
    #clf = linear_model.Lasso(alpha=0.1)
    clf = linear_model.RANSACRegressor()
    if final_x.shape[0] == 0:
        print route,twn,time
    else:
        print 'has training data',route,twn,time,final_x.shape[0]
        clf.fit(final_x[:,0:6],final_x[:,6])
    return clf

def get_lasso(data_origin,test_data,route,twn,day,time):
    clf = LASSO(data_origin,route,twn,time) ##用原始数据学模型
    return clf.predict(test_data[route][day][twn-6:twn]) ##用整合数据做预测
    #return 0


##加载测试数据前6窗数据
def load_test_data():
    df_test=pd.read_csv('../../dataSets/training/0428/my_test_20min_avg_travel_time.csv')
    #df_test=pd.read_csv('../../dataSets/testing_phase1/trajectories_20min_avg_travel_time.csv')
    trajectories_test = {}
    start_time  = datetime.strptime("2016-10-11 00:00:00",'%Y-%m-%d %H:%M:%S')

    for idx in range(df_test.shape[0]):
        line = df_test.iloc[idx]
        route = str(line['intersection_id']+str(line['tollgate_id']))
        stime = datetime.strptime(line['time_window'][1:20],'%Y-%m-%d %H:%M:%S')
        day = (stime - start_time).days
        tw_n = (stime.hour * 60 + stime.minute)/20

        if route not in trajectories_test.keys():
            trajectories_test[route] = np.zeros(72*7).reshape(7,72)
        trajectories_test[route][day][tw_n] = line['avg_travel_time']

    #print trajectories_test['A2'][0]
    for route in trajectories_test.keys():
        trajectories_test[route][:,0:18] = 0;
        trajectories_test[route][:,30:45] = 0;
        trajectories_test[route][:,57:72] = 0;

    return trajectories_test

def predict_lasso():
    '''
    加载数据，前6个窗来自于加载的测试数据，后六个窗来自于均值数据
    输出：Lasso预测的结果
    '''
    data1 = load_test_data()  #加载的testdata只有前6个窗
    data2 = predict_mean()
    data_origin = load_train_data()
    data = {}
    #for route in data1.keys():
    #    data[route] = data1[route]+data2[route] ##整合后的数据供Lasso预测使

    #对于输入中有0的用均值代替
    for route in data1.keys():
        data[route] = np.zeros(7*72).reshape(7,72)

    tw_list = [18,19,20,21,22,23,24,25,26,27,28,29,45,46,47,48,49,50,51,52,53,54,55,56]
    for route in data1.keys():
        for day in range(7):
            for twn in tw_list:
                if data1[route][day][twn] == 0:
                    data[route][day][twn] = data2[route][day][twn]
                else:
                    data[route][day][twn] = data1[route][day][twn]

    # 预测
    pstart_time=datetime.strptime("2016-10-11 00:00:00",'%Y-%m-%d %H:%M:%S')
    y_tw_list = [24,25,26,27,28,29,51,52,53,54,55,56]
    #y_tw_list = [18,19,20,21,22,23,45,46,47,48,49,50]
    #get_lasso(data_origin,data,'A2',24,0,pstart_time+timedelta(days=0))
    pred_data = {}
    for route in ['A2','A3','B3']:
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
        start_time = datetime.strptime('2016-10-11 08:00:00',"%Y-%m-%d %H:%M:%S")
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
        start_time = datetime.strptime('2016-10-11 17:00:00',"%Y-%m-%d %H:%M:%S")
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


def score(test_data,predict_data):
    '''
    test_data:含实际label
    predict_data:预测的值
    '''
    tw = [24,25,26,27,28,29,51,52,53,54,55,56]
    p = predict_data
    x = test_data
    #print p

    summ = 0.0
    n = 0.0

    for route in predict_data.keys():
        for tw_n in tw:
            for day in range(7):
                if x[route][day][tw_n] != 0:
                    xxx = x[route][day][tw_n]
                    summ += np.abs((xxx - p[route][day][tw_n])/xxx)
                    n += 1
    print summ,n
    e = summ / n
    print e


#print load_test_data()
## 学习
pred_mean_data = predict_mean()
pred_lasso_data = predict_lasso()
#lasso预测的值偏大!

print pred_mean_data
print pred_lasso_data

pred_data = {}
y_tw_list = [24,25,26,27,28,29,51,52,53,54,55,56]
#print pred_lasso_data.keys()
for route in pred_mean_data.keys():
    pred_data[route] = np.zeros(7*72).reshape(7,72).astype('Float64')

for route in ["A2","A3","B3"]:
    for day in range(7):
        for twn in y_tw_list:
            if pred_lasso_data[route][day][twn] > 0:
                #print (pred_mean_data[route][day][twn] + pred_lasso_data[route][day][twn])/2.0
                pred_data[route][day][twn] = (2*pred_mean_data[route][day][twn] + 0*pred_lasso_data[route][day][twn])/2.0 ##两种方法结果取均值
            else:
                pred_data[route][day][twn] = pred_mean_data[route][day][twn]
for route in ["B1","C3","C1"]:
    pred_data[route] = pred_mean_data[route]

test_data = load_test_data()
score(pred_data,test_data)
out_data(pred_data)
