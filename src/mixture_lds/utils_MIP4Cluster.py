import numpy as np
import pandas as pd
from scipy.io import arff
import random
import re
import os

class utils_MIP4Cluster():
      
    def __init__(self):
        super().__init__()

    def summary_reg(self, folder_path):
        data_dict = {
            "validation": {},
            "countvalidation": {},
            "std": {},
            "mean": {},
            "duration": {}
        }

        pattern = re.compile(r'_(\d+(?:\.\d+)?)\.npy$')
        for filename in os.listdir(folder_path):
            if filename.endswith(".npy"):
                file_path = os.path.join(folder_path, filename)
                match = pattern.search(filename)
                if match:
                    key = match.group(1)
                    if "validation" in filename:
                        data_dict["validation"][key] = np.load(file_path)
                        data_dict["countvalidation"][key] = sum(1 for x in data_dict["validation"][key] if x != 0)
                    elif "std" in filename:
                        data_dict["std"][key] = np.load(file_path)
                    elif "mean" in filename:
                        data_dict["mean"][key] = np.load(file_path)
                    elif "duration" in filename:
                        data_dict["duration"][key] = np.load(file_path)

        for category in data_dict:
            data_dict[category] = dict(sorted(data_dict[category].items()))
        for category, values in data_dict.items():
            print(f"\nCategory: {category} ")
            for key, data in values.items():
                print(f"{key}: {data}")

    def dynamic_generate(self, g,f_dash,proc_noise_std,obs_noise_std,inputs,T):
        import sys
        sys.path.append('../')
        from typing import List
        from copy import deepcopy
        import numpy as np
        import pandas as pd
        from math import sqrt
        from sklearn.preprocessing import scale, normalize, MinMaxScaler
        from scipy.spatial.distance import pdist, squareform
        from scipy.io import arff
        from Mip4cluster.src.mixture_lds.inputlds import dynamical_system

        import matplotlib.pyplot as plt
        """
        parametrs for dynamical_system(A,B,C,D, **kwargs)
        input: A,B,C,D, **kwargs
        phi_(t+1) = A*phi_t + B*X_t + w_(t+1)
            Y_t = C*phi_t + D*X_t + v_t
        A--> g:[n,n] 
        B--> B:[n,d] matrix for inputs, d is the dimension of inputs
        C--> f_dash:[m,n] 
        D--> D:[m,d] matrix for inputs, d is the dimension of inputs
        n is the dimension of hidden states (phi: [T,n]);
        m is the dimension of observations (Y: [T,m]);
        d is the dimension of inputs (X: [T,d]).
        
        """
        n=len(g)
        m=len(f_dash)
        if inputs == 0: # no inputs
            inputs = np.zeros((m,T))
        dim = len(inputs) # dimension of inputs
        ds1 = dynamical_system(g,np.zeros((n,dim)),f_dash,np.zeros((m,dim)),
                process_noise='gaussian',
                observation_noise='gaussian', 
                process_noise_std=proc_noise_std, 
                observation_noise_std=obs_noise_std)

        h0=np.ones(ds1.d) # initial state
        ds1.solve(h0=h0, inputs=inputs, T=T)
        return np.asarray(ds1.outputs).reshape(T,m) #.tolist()
        
    def data_generation(self, g,f_dash,pro_rang,obs_rang,T,S):
        proL=len(pro_rang)
        obsL=len(obs_rang)
        file_name =[]
        for gg in range(len(g)):
            n=gg+2
            m=2
            cluster_1 = []
            cluster_2 = []
            for i in range(proL):
                for j in range(obsL):
                    proc_noise_std=pro_rang[i]
                    obs_noise_std=obs_rang[j]
                    # Generate data
                    # inputs = np.zeros((2,T))
                    inputs = 0
                    for k in range(S):
                        data_1 = self.dynamic_generate(g[gg, 0],f_dash[gg][0],proc_noise_std,obs_noise_std,inputs,T)
                        cluster_1.append(data_1)
                        
                        data_2 = self.dynamic_generate(g[gg, 1],f_dash[gg][1],proc_noise_std,obs_noise_std,inputs,T)
                        cluster_2.append(data_2)

            Y = np.concatenate((np.array(cluster_1),np.array(cluster_2)),axis=0)
            Y_label = np.concatenate((np.zeros(len(cluster_1)),np.ones(len(cluster_2))),axis=0)
            print(Y.shape, Y_label.shape)
            data = Y.reshape(320,-1)
            #label = Y_label.reshape(32,)
            f_name = f'./data/{n}_{m}_test.npy'
            with open(f_name, 'wb') as f:
                np.save(f, data)
            file_name.append(f_name)
        return file_name

    def datacleaning(self, data_dir, name, S, I, T, M, J):
        # 3,10,32,20,2
        if name =='lds':
            path_list = ['./data/raw/2_2_test.npy', './data/raw/3_2_test.npy','./data/raw/4_2_test.npy']
            ############################# data #####################
            X__=[]
            for path in path_list:
                data_X = np.load(path)
                data_X = data_X.reshape(2*S, I, T, M)
                #print(data_X.shape)

                X_ = np.zeros((S, 2*I, T, M))
                for s in range(S):
                    xx_1 = data_X[s]
                    xx_2 = data_X[s+S]
                    X_[s] = np.concatenate((xx_1,xx_2),axis=0).reshape(2*I, T, M)
                X__.append(X_)
            X = np.array(X__).reshape(3, S, 2*I, T, M) #3, 10, 32, 20, 2

            label=np.concatenate((np.zeros(I),np.ones(I)),axis=0)

            return X,label

        # Test for different t=2,3,4,5,6 or 30,60,90,120,140
        elif name =='ecg':
            def a2p(path):
                data, meta = arff.loadarff(path)
                df = pd.DataFrame(data)
                return df

            trainpath = "./data/raw/ECG5000_TRAIN.arff"
            X_data = a2p(trainpath)
            print(X_data.target.value_counts())
            X_data.head()
            X_1 = X_data[X_data.target==b'1'].iloc[:,:-1].values
            X_2 = X_data[X_data.target==b'2'].iloc[:,:-1].values
            idx_1 = np.arange(len(X_1))
            idx_2 = np.arange(len(X_2))
            iidx_1 = random.sample(sorted(idx_1), I)
            iidx_2 = random.sample(sorted(idx_2), I)
            np.random.shuffle(iidx_1)
            np.random.shuffle(iidx_2)
            XX_1 = X_1[iidx_1]
            XX_2 = X_2[iidx_2]
            X = np.concatenate((XX_1, XX_2), axis=0).reshape(S, 2*I,T,M)#1*S*T*M=1*30*140*1
            label=np.concatenate((np.zeros(I),np.ones(I)),axis=0)
            return X, label

        elif data_dir!=[]:
            data_dict = {}
            for file_name in os.listdir(data_dir):
                if file_name.endswith(".npy"):  # 只处理 .npy 文件
                    file_path = os.path.join(data_dir, file_name)
                    key_name = file_name.replace('.npy', '')  # 去掉后缀作为键
                    data_dict[key_name] = np.load(file_path, allow_pickle=True)
                    if len(data_dict[key_name].shape) ==3:
                        S, T, M = data_dict[key_name].shape
                    elif len(data_dict[key_name].shape) ==4:
                        S, I_, T, M = data_dict[key_name].shape
                    else:
                        print(f"Error processing {file_name}")

            if data_dict.shape ==(2, 3):
                Y = np.concatenate((np.array(data_dict[0]),np.array(data_dict[1])),axis=0)
                Y_label = np.concatenate((np.zeros(len(data_dict[0])),np.ones(len(data_dict[1]))),axis=0)
                print(Y.shape, Y_label.shape)
                data = Y.reshape(S*I*2,-1)
                with open('{name}_test.npy', 'wb') as f:
                    np.save(f, data)

            elif data_dict.shape ==(4, 1):
                label=np.concatenate((np.zeros(I_/2),np.ones(I_/2)),axis=0)
                
            else:
                print("Cannot deal with the given data type, please contact owner!")

        else:
            print("Wrong data type!")

    def plot_MIF4cluster_methods(self, path, methods, name, cutdown ):
        import numpy as np
        import matplotlib.pyplot as plt
        from mpl_toolkits.axisartist.parasite_axes import HostAxes, ParasiteAxes
        from matplotlib.backends.backend_pdf import PdfPages

        colormap=('#4292c6','#696969', '#CD5C5C' ,'#FFD700', '#6B8E23')
        labels=methods
        # load results on Synthetic dataset
        mean = []
        std = []
        for method in methods:
            if cutdown==True:
                mean.append(np.load(f"{path}f1_{method}_{name}_mean_cd.npy"))
                std.append(np.load(f"{path}f1_{method}_{name}_std_cd.npy"))
            else:
                mean.append(np.load(f"{path}f1_{method}_{name}_mean.npy"))
                std.append(np.load(f"{path}f1_{method}_{name}_std.npy"))
        mean = np.concatenate(mean).reshape(5,3) # Concatenate along axis 0 by default
        std = np.concatenate(std).reshape(5,3) # Concatenate along axis 0 by default
        print([mean.shape, std.shape])
        ##############################################
        ############# plot for n = 2,3,4 #############
        ########## fft, dtw, ind, emh, gurobi ########
        ##############################################

        met_range=[*range(5)]
        nx_range=np.arange(2,5)

        # fig = plt.figure(figsize=(10,6), dpi=100)
        # mean = np.array(results[f"{method}_{name}_mean"] for method in methods)
        # std = np.array(results[f"{method}_{name}_std"] for method in methods)
        fig, ax = plt.subplots(figsize=(8,5), dpi=200)
        width=0.1
        for m in met_range:
            x=nx_range+m*width
            y=mean[m,:]
            y_error=1.96*std[m,:]/np.sqrt(50)
            ax.plot(x,y, color=colormap[m],label=labels[m])
            ax.errorbar(x, y, yerr = y_error, fmt ='.', color=colormap[m],capsize=4,capthick=2) #,align='edge'

        # plt.legend(bbox_to_anchor=(1.02,1.25),fontsize=14,frameon=False,ncol=2)
        plt.legend(fontsize=14,frameon=False,ncol=1,loc='lower right')
        plt.yticks(ticks=[0.0,0.25,0.5,0.75,1.0],labels=[0.0,0.25,0.5,0.75,1.0],fontsize=14)
        plt.xticks(ticks=[2.3,3.3,4.3],labels=[2,3,4],fontsize=14)
        plt.xlabel('dimensions of system matrices '+r'$n$',fontsize=16)
        plt.ylabel('F1 score',fontsize=16)
        plt.savefig(f'./reports/figures/{name}_f1.png', bbox_inches='tight')
