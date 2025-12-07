import pandas as pd
import random
import numpy as np

def merge_file(file1,file2,file3):
    df1 = pd.read_csv(file1, usecols=['hidden_size','num_layers','dropout','seq_length','avg_loss'])
    df2 = pd.read_csv(file2, usecols=['hidden_size','num_layers','dropout','seq_length','avg_loss'])
    
    merged_df = pd.concat([df1, df2], ignore_index=True)
    
    merged_df.to_csv(file3, index=False)

def reduce_size(file1):
    
    # Read the first 20 rows along with header
    df = pd.read_csv(file1, nrows=20)
    
    # Save to new CSV with the same header
    df.to_csv(file1, index=False)


def choose_par(file1):
    df = pd.read_csv(file1)
    par1=random.randint(0,len(df)-1)
    par2=random.randint(0,len(df)-1)
    if df.at[par1,'front']<df.at[par2,'front']:
        return par1
    if df.at[par1,'front']>df.at[par2,'front']:
        return par2
    if df.at[par1,'dist']>df.at[par2,'dist']:
        return par1
    if df.at[par1,'dist']<df.at[par2,'dist']:
        return par2
    return par1

def crossover(i,j,file1):
    df = pd.read_csv(file1)
    low=min(df.at[i,'hidden_size'],df.at[j,'hidden_size'])
    high=max(df.at[i,'hidden_size'],df.at[j,'hidden_size'])
    h=random.randint(low,high)
    low=min(df.at[i,'num_layers'],df.at[j,'num_layers'])
    high=max(df.at[i,'num_layers'],df.at[j,'num_layers'])
    n=random.randint(low,high)
    d=random.random()*(df.at[i,'dropout']-df.at[j,'dropout'])+df.at[j,'dropout']
    low=min(df.at[i,'seq_length'],df.at[j,'seq_length'])
    high=max(df.at[i,'seq_length'],df.at[j,'seq_length'])
    s=random.randint(low,high)
    return [h,n,d,s]

def mutate(l):
    if random.random()<0.17:
        l[0]=random.randint(32,256)
    if random.random()<0.17:
        l[1]=random.randint(1,6)
    if random.random()<0.17:
        l[2]=(random.random()/2)
    if random.random()<0.17:
        l[3]=random.randint(5,100)
    return l
def new_child(file1):
    p1=choose_par(file1)
    p2=choose_par(file1)
    return mutate(crossover(p1,p2,file1))


def update_file(file1,file2):
    df = pd.read_csv(file1)
    h=df['hidden_size']
    s=df['seq_length']
    n=df['num_layers']
    df['size']=4*(s*h+h*h+h+(n-1)*(2*h*h+h))+h+1
    df = df[['hidden_size','num_layers','dropout','seq_length','avg_loss','size']]
    df['front'] = 0
    k=0
    while ((df['front'] == 0).any()):
        k+=1
        for idx1,rows1 in df.iterrows():
            if df.at[idx1,'front']>0:
                continue
            for idx2,rows2 in df.iterrows():
                if idx1==idx2:
                    continue
                if df.at[idx2,'size']<=df.at[idx1,'size'] and df.at[idx2,'avg_loss']<df.at[idx1,'avg_loss'] and df.at[idx2,'front']<=0:
                    df.at[idx1,'front']=-1
                    break
        df['front'] = df['front'].replace(0, k)
        df['front'] = df['front'].replace(-1, 0)
    df = df.sort_values(by='front').reset_index(drop=True)
    print(len(df))
    last_front=(df.at[len(df)-1,'front'])
    df['dsize']=0.0
    df['dloss']=0.0
    for ii in range(1,last_front+1):
        A=[]
        for idx,rows in df.iterrows():
            if (df.at[idx,'front'])==ii:
                A.append(idx)
        print(A)
        subset = df.iloc[A[0]:A[-1]+1].sort_values(by='size')
        df.iloc[A[0]:A[-1]+1] = subset.values
        df.reset_index(drop=True, inplace=True)
        
        for i in range(len(A)):
            if i==0 or i==len(A)-1:
                df.at[A[i],'dsize']=10000.0
            elif df.at[A[i],'size']==df.at[A[0],'size'] or df.at[A[i],'size']==df.at[A[-1],'size']:
                df.at[A[i],'dsize']=10000.0
            else:
                print(A[i])
                df.at[A[i],'dsize']=(df.at[A[i+1],'size']-df.at[A[i-1],'size'])/(df.at[A[-1],'size']-df.at[A[0],'size'])
    for ii in range(1,last_front+1):
        A=[]
        for idx,rows in df.iterrows():
            if (df.at[idx,'front'])==ii:
                A.append(idx)
        print(A)
        subset = df.iloc[A[0]:A[-1]+1].sort_values(by='avg_loss')
        df.iloc[A[0]:A[-1]+1] = subset.values
        df.reset_index(drop=True, inplace=True)
        
        for i in range(len(A)):
            if i==0 or i==len(A)-1:
                df.at[A[i],'dloss']=20000.0
            elif df.at[A[i],'avg_loss']==df.at[A[0],'avg_loss'] or df.at[A[i],'avg_loss']==df.at[A[-1],'avg_loss']:
                df.at[A[i],'dloss']=20000.0
            else:
                print(A[i])
                df.at[A[i],'dloss']=(df.at[A[i+1],'avg_loss']-df.at[A[i-1],'avg_loss'])/(df.at[A[-1],'avg_loss']-df.at[A[0],'avg_loss'])
    df['dist']=0.25*df['dsize']+0.75*df['dloss']

    for ii in range(1,last_front+1):
        A=[]
        for idx,rows in df.iterrows():
            if (df.at[idx,'front'])==ii:
                A.append(idx)
        print(A)
        subset = df.iloc[A[0]:A[-1]+1].sort_values(by='dist',ascending=False)
        df.iloc[A[0]:A[-1]+1] = subset.values
        df.reset_index(drop=True, inplace=True)


    df = df[['hidden_size','num_layers','dropout','seq_length','avg_loss','front','dist']]
    df.to_csv(file2, index=False)

