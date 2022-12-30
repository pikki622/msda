class FeatureSelection:

    time_min = globals()
    

    # function to find variation ( like Increase/Inc, Decrease/Dec, Equal/Eql) in each column from previous values
    def count_trend(self):
        inc, dec, eql, j = 0, 0, 0, 0
        while j != len(self) - 1:
            if self[j] > self[j + 1]:
                dec+=1
            elif self[j] < self[j + 1]:
                inc+=1
            else:
                eql+=1
            j+=1
        return {'Inc':inc, 'Dec':dec, 'Eql':eql}
    
    # fuctn to find how each column value changes wrt other column values
    def rate(self, str1, str2):
        import numpy as np

        ind = globals()
        for i in range(2, len(self.columns)):
            ind['ind_{0}'.format(i)] = {str(j):0 for j in range(2, len(df.columns))}
        '''
        ind_2={'2':0,'3':0,'4':0,'5':0,'6':0,'7':0}
        ind_3={'2':0,'3':0,'4':0,'5':0,'6':0,'7':0}
        ind_4={'2':0,'3':0,'4':0,'5':0,'6':0,'7':0}
        ind_5={'2':0,'3':0,'4':0,'5':0,'6':0,'7':0}
        ind_6={'2':0,'3':0,'4':0,'5':0,'6':0,'7':0}
        ind_7={'2':0,'3':0,'4':0,'5':0,'6':0,'7':0}
        '''

        trend = []
        i=0
        av = self.to_numpy()
        while i!=len(av)-1:
            vg=[]
            for k, v in zip(av[i], av[i+1]):
                if k==v:
                    vg.append('Eq')
                elif k>v:
                    vg.append('Dec')
                elif k<v:
                    vg.append('Inc')
            i+=1
            trend.append(vg)
        mat = np.array(trend)
        print(mat)


        i=0
        while i!=len(mat)-1:
            if mat[i+1][2]==str1:
                ind_2['2']+=1
                if mat[i+1][3]==str2:
                    ind_2['3']+=1
                elif mat[i+1][4]==str2:
                    ind_2['4']+=1
                elif mat[i+1][5]==str2:
                    ind_2['5']+=1
                elif mat[i+1][6]==str2:
                    ind_2['6']+=1
                elif mat[i+1][7]==str2:
                    ind_2['7']+=1

            elif mat[i+1][3]==str1:
                ind_3['3']+=1
                if mat[i+1][2]==str2:
                    ind_3['2']+=1
                elif mat[i+1][4]==str2:
                    ind_3['4']+=1
                elif mat[i+1][5]==str2:
                    ind_3['5']+=1
                elif mat[i+1][6]==str2:
                    ind_3['6']+=1
                elif mat[i+1][7]==str2:
                    ind_3['7']+=1

            elif mat[i+1][4]==str1:
                ind_4['4']+=1
                if mat[i+1][2]==str2:
                    ind_4['2']+=1
                elif mat[i+1][3]==str2:
                    ind_4['3']+=1
                elif mat[i+1][5]==str2:
                    ind_4['5']+=1
                elif mat[i+1][6]==str2:
                    ind_4['6']+=1
                elif mat[i+1][7]==str2:
                    ind_4['7']+=1

            elif mat[i+1][5]==str1:
                ind_5['5']+=1
                if mat[i+1][2]==str2:
                    ind_5['2']+=1
                elif mat[i+1][3]==str2:
                    ind_5['3']+=1
                elif mat[i+1][4]==str2:
                    ind_5['4']+=1
                elif mat[i+1][6]==str2:
                    ind_5['6']+=1
                elif mat[i+1][7]==str2:
                    ind_5['7']+=1

            elif mat[i+1][6]==str1:
                ind_6['6']+=1
                if mat[i+1][2]==str2:
                    ind_6['2']+=1
                elif mat[i+1][3]==str2:
                    ind_6['3']+=1
                elif mat[i+1][4]==str2:
                    ind_6['4']+=1
                elif mat[i+1][5]==str2:
                    ind_6['5']+=1
                elif mat[i+1][7]==str2:
                    ind_6['7']+=1

            elif mat[i+1][7]==str1:
                ind_7['7']+=1
                if mat[i+1][2]==str2:
                    ind_7['2']+=1
                elif mat[i+1][3]==str2:
                    ind_7['3']+=1
                elif mat[i+1][4]==str2:
                    ind_7['4']+=1
                elif mat[i+1][5]==str2:
                    ind_7['5']+=1
                elif mat[i+1][6]==str2:
                    ind_7['6']+=1           
            i+=1
        #return {'2':ind_2, '3':ind_3, '4':ind_4, '5':ind_5, '6':ind_6, '7':ind_7}
        return {str(k):ind['ind_{0}'.format(i)] for i, k in zip(range(2, len(df.columns)),range(2, len(df.columns)))}
        #return {str(k):ind['ind_{0}'.format(i) for i in range(2, len(df.columns))]  for k in range(2, len(df.columns))} ind['ind_{0}'.format(i)] = {str(j):0 for j in range(2, len(df.columns))}
    
    # Function To Calculate Ratios For Each Variation In Each Column Value wrt Other
    # Column Value and Finding Max Ratio To Establish Change of a given Column Value wrt Other Column Value.

    def ratio(self, column):
        columns = list(self.columns)
        ind = columns.index(column)

        cases = [('Inc','Inc'),('Inc','Dec'),('Inc','Eq'),
           ('Dec','Dec'),('Dec','Inc'),('Dec','Eq'),
           ('Eq','Eq'),('Eq','Dec'),('Eq','Inc')]

        ratios, index_pairs = [], []
        for i in cases:
            j = 0
            dict = FeatureSelection.rate(self, i[0], i[1])[str(ind)]
            v1 = dict.pop(str(ind))
            v2 = max(list(dict.values()))
            pairs = list(dict.items())
            for i in pairs:
                if i[1]==v2:
                    j=int(i[0])
            ratios.append(v2/v1)
            index_pairs.append((j, ind))

        rat = max(ratios)
        indx = index_pairs[ratios.index(rat)]

        print(f'For Sensor Column:- {columns[ind]}')
        print('Ratio is:', rat)
        print(
            f"When Sensor Column \'{columns[ind]}\' values are {cases[ratios.index(rat)][0]} , Sensor Column \'{columns[j]}\' values are {cases[ratios.index(rat)][1]}"
        )



    # Function to detect Avg and Standard Deviation for some
    # window accross each sensor column

    def window(self):
        import numpy as np
        import operator, statistics
        from statistics import mean
        time = int(input('Enter Time in Minutes for the Window: (Must be a Multiple of 2):'))
        index = int(time/2+1)
        col_avg, col_std = [], []
        #cols = list(df.columns)
        #cols = cols[2:]
        cols = list(self.columns)[2:]
        #columns = df.iloc[:,2:].values.tolist()
        columns = (
            self.iloc[:, 2:]
            .aggregate(lambda x: [x.tolist()], axis=0)
            .map(lambda x: x[0])
        )
        #columns=[Light,Temp_Air,Temp_Soil,Soil_Moisture_1,Soil_Moisture_2,Soil_Moisture_3]

        for i in columns:
            avg, std = [], []
            for j in range(0, len(i), index):
                avg.append(np.mean(i[j:j+index]))
                std.append(statistics.stdev(i[j:j+index]))
            col_avg.append(np.mean(avg))
            col_std.append(np.mean(std))

        for i in range(len(cols)):
            print(
                f'Rate of Change of AVG Across Window For Sensor Column {cols[i]}: {col_avg[i]}'
            )
            print(
                f'Rate of Change of STD Across Window For Sensor Column {cols[i]}: {col_std[i]}'
            )

    # Function to generate features for each sensor columns
    def features(self, start_sensor_column_index):
        import numpy as np
        import pandas as pd
        self.iloc[:, start_sensor_column_index:len(self.columns)].astype(np.float64)
        #print(df.columns)
        #print(df.head())
        change_ovr_time = globals()
        av = self.to_numpy()
        #for i in range(2, len(df.columns)):
        change_ovr_time = {str(j):[] for j in df.columns[2:]} #df.columns[2:]
        #print(change_ovr_time)
        '''
            change_ovr_time={'Light':[],'Temp_Air':[],'Temp_Soil':[],'Soil_Moisture_1':[],
                                 'Soil_Moisture_2':[],'Soil_Moisture_3':[]}
        '''
        i = 0
        trend = []
        while i!=len(av)-1: #-1
            vg = [(v-k)/2 for k, v in zip(av[i][2:], av[i+1][2:])]
            i+=1
            trend.append(vg)
        array = np.array(trend)
        #print(len(array))
        change_ovr_time = {str(j):list(array[:,i]) for i, j in zip(range(array.shape[1]), df.columns[2:])} #df.columns[2:]
        '''
            for i in range(2, len(df.columns)):
                    change_ovr_time = {str(j):[] for j in df.columns[2:]}
                
            change_ovr_time ={'Light':list(array[:,0]),'Temp_Air':list(array[:,1]),'Temp_Soil':list(array[:,2]),'Soil_Moisture_1':list(array[:,3]),
                                 'Soil_Moisture_2':list(array[:,4]),'Soil_Moisture_3':list(array[:,5])}
        '''
        rate_of_change_ovr_time={}
        i=0
        trend=[]
        while i!=len(array)-1: #-1
            vg = [(v-k)/2 for k, v in zip(array[i],array[i+1])]
            i+=1
            trend.append(vg)
        array=np.array(trend)
        rate_of_change_ovr_time = {str(j):list(array[:,i]) for i, j in zip(range(array.shape[1]), df.columns[2:])} #df.columns[2:]
        '''
                for i in range(2, len(df.columns)):
                    change_ovr_time = {str(j):[] for j in df.columns[2:]}
                
                rate_of_change_ovr_time={'Light':list(array[:,0]),'Temp_Air':list(array[:,1]),'Temp_Soil':list(array[:,2]),'Soil_Moisture_1':list(array[:,3]),
                                 'Soil_Moisture_2':list(array[:,4]),'Soil_Moisture_3':list(array[:,5])}
        '''
        growth_decay={}
        i=0
        trend=[]
        while i!=len(av)-1: #-1
            vg=[]
            for k,v in zip(av[i][2:],av[i+1][2:]):
                #vg.append((v-k)/k)

                try:
                    vg.append((v-k)/k)
                except ZeroDivisionError:
                    vg.append(0)

            i+=1
            trend.append(vg)
        array=np.array(trend)
        growth_decay = {str(j):list(array[:,i]) for i, j in zip(range(array.shape[1]), df.columns[2:])} #df.columns[2:]
        '''
                for i in range(2, len(df.columns)):
                    change_ovr_time = {str(j):[] for j in df.columns[2:]}
                
                growth_decay={'Light':list(array[:,0]),'Temp_Air':list(array[:,1]),'Temp_Soil':list(array[:,2]),'Soil_Moisture_1':list(array[:,3]),
                                 'Soil_Moisture_2':list(array[:,4]),'Soil_Moisture_3':list(array[:,5])}
        '''
        rate_growth_decay={}
        i=0
        trend=[]
        while i!=len(array)-1: #-1
            vg = [(v-k)/2 for k, v in zip(array[i],array[i+1])]
            i+=1
            trend.append(vg)
        array=np.array(trend)
        rate_growth_decay = {str(j):list(array[:,i]) for i, j in zip(range(array.shape[1]), df.columns[2:])} #df.columns[2:]
        threshold_growth_decay = {str(j):np.mean(growth_decay[str(i)]) for i, j in zip((growth_decay.keys()), df.columns[2:])} #df.columns[2:]
        #print(f' threshold growth decay dict: {threshold_growth_decay}')

        count_decay_growth={}
        counts = [
            sum(val >= j for val in i)
            for i, j in zip(
                list(growth_decay.values()), list(threshold_growth_decay.values())
            )
        ]
        count_decay_growth = {str(j):counts[i] for i,j in zip(range(len(counts)), df.columns[2:])} #df.columns[2:]

        '''

        for i in range(len(counts)):
            count_decay_growth = {str(j):counts[i] for j in df.columns[2:]}
        '''


        #    print(f' count_decay_growth dict: {count_decay_growth}')  

        df_change_ovr_time = {str(j):list(change_ovr_time.values())[i] \
                                  for i, j in zip(range(len(change_ovr_time.keys())), df.columns[2:])} # df.columns[2:]

        df_rate_of_change_ovr_time = {str(j):list(rate_of_change_ovr_time.values())[i] \
                                  for i, j in zip(range(len(rate_of_change_ovr_time.keys())), df.columns[2:])} # df.columns[2:]

        df_growth_decay = {str(j):list(growth_decay.values())[i] \
                                  for i, j in zip(range(len(growth_decay.keys())), df.columns[2:])} # df.columns[2:]

        df_rate_growth_decay = {str(j):list(rate_growth_decay.values())[i] \
                                  for i, j in zip(range(len(rate_growth_decay.keys())), df.columns[2:])} # df.columns[2:]


        '''
        for i, j in zip(list(change_ovr_time.values()), df.columns[2:]):
            df_change_ovr_time = {str(j):list(change_ovr_time.values())[i] for j in df.columns[2:]}
            
        for i, j in zip(list(rate_of_change_ovr_time.values()), df.columns[2:]):
            df_rate_of_change_ovr_time = {str(j):list(rate_of_change_ovr_time.values())[i] for j in df.columns[2:]}

        for i, j in zip(list(growth_decay.values()), df.columns[2:]):
            df_growth_decay = {str(j):list(growth_decay.values())[i] for j in df.columns[2:]}

        for i, j in zip(list(rate_growth_decay.values()), df.columns[2:]):
            df_rate_growth_decay = {str(j):list(rate_growth_decay.values())[i] for j in df.columns[2:]}

           
                for i in range(2, len(df.columns)):
                    df_change_ovr_time = {str(j):[] for j in df.columns[2:]}


                    df_rate_of_change_ovr_time = {str(j):[] for j in df.columns[2:]}
                    df_growth_decay = {str(j):[] for j in df.columns[2:]}
                    df_rate_growth_decay = {str(j):[] for j in df.columns[2:]}
                
                
                df_change_ovr_time={'Light':list(change_ovr_time.values())[0],'Temp_Air':list(change_ovr_time.values())[1],'Temp_Soil':list(change_ovr_time.values())[2],
                                    'Soil_Moisture_1':list(change_ovr_time.values())[3],'Soil_Moisture_2':list(change_ovr_time.values())[4],'Soil_Moisture_3':list(change_ovr_time.values())[5]}
                df_rate_of_change_ovr_time={'Light':list(rate_of_change_ovr_time.values())[0],'Temp_Air':list(rate_of_change_ovr_time.values())[1],'Temp_Soil':list(rate_of_change_ovr_time.values())[2],
                                            'Soil_Moisture_1':list(rate_of_change_ovr_time.values())[3],'Soil_Moisture_2':list(rate_of_change_ovr_time.values())[4],'Soil_Moisture_3':list(rate_of_change_ovr_time.values())[5]}
                df_growth_decay={'Light':list(growth_decay.values())[0],'Temp_Air':list(growth_decay.values())[1],'Temp_Soil':list(growth_decay.values())[2],
                                 'Soil_Moisture_1':list(growth_decay.values())[3],'Soil_Moisture_2':list(growth_decay.values())[4],'Soil_Moisture_3':list(growth_decay.values())[5]}
                df_rate_growth_decay={'Light':list(rate_growth_decay.values())[0],'Temp_Air':list(rate_growth_decay.values())[1],'Temp_Soil':list(rate_growth_decay.values())[2],
                                      'Soil_Moisture_1':list(rate_growth_decay.values())[3],'Soil_Moisture_2':list(rate_growth_decay.values())[4],'Soil_Moisture_3':list(rate_growth_decay.values())[5]}
        '''
        df1= pd.DataFrame(df_change_ovr_time,columns=list(df_change_ovr_time.keys()))
        df1.to_csv('features_change_over_time.csv',index=False)
        df2= pd.DataFrame(df_rate_of_change_ovr_time,columns=list(df_change_ovr_time.keys()))
        df2.to_csv('features_rate_of_change_over_time.csv',index=False)
        df3= pd.DataFrame(df_growth_decay,columns=list(df_change_ovr_time.keys()))
        df3.to_csv('features_growth_decay.csv',index=False)
        df4= pd.DataFrame(df_rate_growth_decay,columns=list(df_change_ovr_time.keys()))
        df4.to_csv('features_rate_growth_decay.csv',index=False)

        print('Count of Growth/Decay value for each Sensor Column Values above or below a threshold value:\n',count_decay_growth)

        return change_ovr_time,rate_of_change_ovr_time,growth_decay,rate_growth_decay,(threshold_growth_decay,count_decay_growth)   



    # Functions to create plot of various sensors and features
    def plot(df, logging_interval):
        import numpy as np
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        import operator, statistics
        from statistics import mean
        #columns=[Light,Temp_Air,Temp_Soil,Soil_Moisture_1,Soil_Moisture_2,Soil_Moisture_3]
        cols = list(df.columns)[2:]
        #columns = df.iloc[:,2:].values.tolist()
        columns =  df.iloc[:,2:].aggregate(lambda x: [x.tolist()], axis=0).map(lambda x:x[0])
        #print(len(columns))
        time_min = []
        initial=0
        for i in range((df.shape[0])):
            time_min.append(initial)
            initial+=logging_interval
        def best_fit_slope_and_intercept(xs,ys):
            m = (((mean(xs)*mean(ys)) - mean(xs*ys)) /
                ((mean(xs)*mean(xs)) - mean(xs*xs)))
            b = mean(ys) - m*mean(xs)
            return m, b

        for i in range(len(columns)):
            m,b=best_fit_slope_and_intercept(np.array(time_min),np.array(columns[i]))
            regression_line = [(m*x)+b for x in np.array(time_min)]
            plt.subplot((len(columns)/2)+1,2,i+1) # plt.subplot(3,2,i+1) plt.subplot(3,(len(columns)/2)+1,i+1)
            plt.scatter(time_min,columns[i], color='red')
            plt.plot(time_min,regression_line,'b-')
            plt.title('Time vs {}'.format(list(df.columns)[i+2]))
            plt.ylabel('{}'.format(list(df.columns)[i+2]))
        #plt.title('Slope For Sensor Columns Showing Max Variation', fontsize=16)
        plt.suptitle('Slope For Sensor Columns Showing Max. Variation', fontsize=16)
        #plt.show()
        plt.savefig('slope_with_max_variation.pdf') #, bbox_inches='tight'    

    # function to estimate change over time & create plot for each sensor/features.
    def plot_change_ovr_time(df, feature, logging_interval):
        import numpy as np
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        import operator, statistics
        from statistics import mean
        columns=list(feature.values())
        time_min = []
        initial=0
        for i in range((df.shape[0])):
            time_min.append(initial)
            initial+=logging_interval
        def best_fit_slope_and_intercept(xs,ys):
            m = (((mean(xs)*mean(ys)) - mean(xs*ys)) /
                ((mean(xs)*mean(xs)) - mean(xs*xs)))
            b = mean(ys) - m*mean(xs)
            return m, b

        for i in range(len(columns)):
            m,b=best_fit_slope_and_intercept(np.array(time_min[1:]),np.array(columns[i]))
            regression_line = [(m*x)+b for x in np.array(time_min[1:])]
            plt.subplot((len(columns)/2)+1,2,i+1) # plt.subplot(2,3,i+1) plt.subplot(3,(len(columns)/2)+1,i+1)
            plt.scatter(time_min[1:],columns[i],color='red')
            plt.plot(time_min[1:],regression_line,'b-')
            plt.title('Time vs {}'.format(list(df.columns)[i+2]))
            plt.ylabel('{}'.format(list(df.columns)[i+2]))

        plt.suptitle('Slope For Sensor Columns Showing Change Over Time feature', fontsize=16)
        #plt.show()
        plt.savefig('slope_with_change_over_time.pdf') #, bbox_inches='tight'

    # function to estimate rate of change over time & create plot across each sensor/features.
    def plot_rate_of_change_ovr_time(df, feature, logging_interval):
        import numpy as np
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        import operator, statistics
        from statistics import mean
        columns=list(feature.values())
        time_min = []
        initial=0
        for i in range((df.shape[0])):
            time_min.append(initial)
            initial+=logging_interval
        def best_fit_slope_and_intercept(xs,ys):
            m = (((mean(xs)*mean(ys)) - mean(xs*ys)) /
                ((mean(xs)*mean(xs)) - mean(xs*xs)))
            b = mean(ys) - m*mean(xs)
            return m, b

        for i in range(len(columns)):
            m,b=best_fit_slope_and_intercept(np.array(time_min[2:]),np.array(columns[i]))
            regression_line = [(m*x)+b for x in np.array(time_min[2:])]
            plt.subplot((len(columns)/2)+1,2,i+1) # plt.subplot(2,3,i+1) plt.subplot(3,(len(columns)/2)+1,i+1)
            plt.scatter(time_min[2:],columns[i],color='red')
            plt.plot(time_min[2:],regression_line,'b-')
            plt.title('Rate of Change of {}'.format(list(df.columns)[i+2]))
            plt.ylabel('{}'.format(list(df.columns)[i+2]))

        plt.suptitle('Slope For Sensor Columns Showing Rate of Change Over Time feature', fontsize=16)
        #plt.show()
        plt.savefig('slope_with_rate_of_change_over_time.pdf') #, bbox_inches='tight'

    # function to estimate growth/decay value & create plot for each sensor/column values
    def plot_growth_decay(df, feature, logging_interval):
        import numpy as np
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        import operator, statistics
        from statistics import mean
        #cols=[Light,Temp_Air,Temp_Soil,Soil_Moisture_1,Soil_Moisture_2,Soil_Moisture_3]
        #cols = list(df.columns)[2:]
        cols =  df.iloc[:,2:].aggregate(lambda x: [x.tolist()], axis=0).map(lambda x:x[0])
        columns=list(feature.values())
        time_min = []
        initial=0
        for i in range((df.shape[0])):
            time_min.append(initial)
            initial+=logging_interval
        def best_fit_slope_and_intercept(xs,ys):
            m = (((mean(xs)*mean(ys)) - mean(xs*ys)) /
                ((mean(xs)*mean(xs)) - mean(xs*xs)))
            b = mean(ys) - m*mean(xs)
            return m, b

        for i in range(len(columns)):
            m,b=best_fit_slope_and_intercept(np.array(cols[i][:-1]),np.array(columns[i]))
            regression_line = [(m*x)+b for x in np.array(cols[i][:-1])]
            plt.subplot((len(columns)/2)+1,2,i+1) # plt.subplot(2,3,i+1) plt.subplot(3,(len(columns)/2)+1,i+1)
            plt.scatter(cols[i][:-1],columns[i],color='red')
            plt.plot(cols[i][:-1],regression_line,'b-')
            plt.title('Growth/Decay for {}'.format(list(df.columns)[i+2]))
            plt.ylabel('Growth/Decay')

        plt.suptitle('Slope For Sensor Columns Showing Growth/Deacy feature', fontsize=16)
        #plt.show()
        plt.savefig('slope_with_Growth_nd_Decay.pdf') #, bbox_inches='tight'

    # function to estimate rate of growth/decay value & create plot for each sensor/column values
    def plot_rate_growth_decay(df, feature, logging_interval):
        import numpy as np
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        import operator, statistics
        from statistics import mean
        columns=list(feature.values())
        time_min = []
        initial=0
        for i in range((df.shape[0])):
            time_min.append(initial)
            initial+=logging_interval
        def best_fit_slope_and_intercept(xs,ys):
            m = (((mean(xs)*mean(ys)) - mean(xs*ys)) /
                ((mean(xs)*mean(xs)) - mean(xs*xs)))
            b = mean(ys) - m*mean(xs)
            return m, b

        for i in range(len(columns)):
            m,b=best_fit_slope_and_intercept(np.array(time_min[2:]),np.array(columns[i]))
            regression_line = [(m*x)+b for x in np.array(time_min[2:])]
            plt.subplot((len(columns)/2)+1,2,i+1) # plt.subplot(2,3,i+1) plt.subplot(3,(len(columns)/2)+1,i+1)
            plt.scatter(time_min[2:],columns[i],color='red')
            plt.plot(time_min[2:],regression_line,'b-')
            plt.title('Growth/Decay Rate for {}'.format(list(df.columns)[i+2]))
            plt.ylabel('Growth/Decay Rate')

        plt.suptitle('Slope For Sensor Columns Showing Growth/Deacy Rate feature', fontsize=16)
        #plt.show()
        plt.savefig('slope_with_rate_growth_decay.pdf') #, bbox_inches='tight'

    # function to count of growth/decay value for each sensor/column values above or below a threshold value.
    def Threshold_Counts(self, feature):
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        cols = list(self.columns)[2:]
        thresholds=list(feature[0].values())
        count=list(feature[1].values())
        plt.plot(cols,thresholds,'go',cols,count,'ro')
        red_patch = mpatches.Patch(color='red', label='Count')
        green_patch = mpatches.Patch(color='green', label='Threshold')
        plt.legend(handles=[red_patch,green_patch])
        plt.xlabel('Sensor Columns')
        plt.ylabel('No. of Counts')
        plt.title('No. of Elements Above Threshold(Mean) Value for Growth/Decay Corresponding to Each Sensor Column')
        #plt.show()
        plt.savefig('Above_Threshold_Count_Growth.pdf') #, bbox_inches='tight'

# In[3]:


class Preprocessing:
    
    # function to find missing values
    def missing(self):
        return sum(self.isnull())

    # function to convert to datetime object, and extract date & time columns
    def datetimeconversion(self, datetimecolindex):
        import pandas as pd
        import datetime
        # convert to datetime object

        self.iloc[:,datetimecolindex] = pd.to_datetime(self.iloc[:,datetimecolindex])

        # Extract date & time separately
        self['Date'] = self.iloc[:,datetimecolindex].dt.date
        self['Time'] = self.iloc[:,datetimecolindex].dt.time

        #Reorder the DATE, TIME column back to BEGINNING index
        cols = list(self)
        # move the column to head of list 
        cols.pop(cols.index('Date')), cols.pop(cols.index('Time'))
        self = self[['Date','Time']+cols[1:]]
        return self

class ExploratoryDataAnalysis:
    
    # profiling your dataset
    def profiling(self): 
        import pandas_profiling as pp
        # To Generate a HTML report file
        html_report = pp.ProfileReport(self, title="Pandas Profiling Report")
        html_report.to_file("./profiling_report.html")
    
    # filtering outliers
    def drop_outliers(self, z_threshold):
        from scipy import stats
        import numpy as np
        constraints = (
            self.select_dtypes(include=[np.number])
            .apply(lambda x: np.abs(stats.zscore(x)) < z_threshold)
            .all(axis=1)
        )
        self.drop(self.index[~constraints], inplace=True)
        return self

    # impute missing values
    def impute(self, modes:int): #start_sensor_column_index, 
        import pandas as pd
        if modes==0:
            #df.iloc[:,start_sensor_column_index:len(df.columns)].fillna(0, inplace=True)
            self.fillna(0, inplace=True)
        elif modes==1:
            #df.iloc[:,start_sensor_column_index:len(df.columns)].fillna(df.mean(), inplace=True)
            self.fillna(self.mean(), inplace=True)
        elif modes==2:
            #df.iloc[:,start_sensor_column_index:len(df.columns)].fillna(df.median(), inplace=True)
            self.fillna(self.median(), inplace=True)
        else:
            #df.iloc[:,start_sensor_column_index:len(df.columns)].fillna(method='bfill', inplace=True) # backfill / bfill: use next valid observation to fill gap.
            self.fillna(method='bfill', inplace=True)
        self.to_csv('df_no_NA.csv')
        return self

    # drop particular column
    def drop_columns(self, column_index_to_drop:int):
        self.drop(self.columns[column_index_to_drop], inplace=True, axis=1)
        return self


class DescriptiveStatistics:

    # Stats for whole dataset
    def desc_all(self, start_sensor_column_index):
        print("Descriptive Statistics")
        return self.iloc[:, start_sensor_column_index:len(self.columns)].describe()
    
    # Stats with relation to response variable and any feature (tobe used if your data is labeled)
    def desc_target_feature(self, target, feature):
        print("Descriptive Statistics by Target-Feature")
        return self.groupby(target)[feature].describe()


class StatisticalEvaluation:

    # Test your data for normality/if the data follows normal distribution
    def normality_test(self, start_sensor_column_index):
        from scipy import stats, norm, shapiro, jarque_bera
        print(
            f"Shapiro-Wilk Normality test (W test statistic, the p-value) =======> {shapiro(self.iloc[:, start_sensor_column_index:len(self.columns)])}  "
        )
        print(
            f"Jarque-Bera Normality test (W test statistic, the p-value) =======> {jarque_bera(self.iloc[:, start_sensor_column_index:len(self.columns)])}"
        )

    
