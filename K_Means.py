# -*- coding: utf-8 -*-

import pandas as pd
from pandas import plotting 
import numpy as np
from numpy import where
import matplotlib.pyplot as plt
%matplotlib inline
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples
from matplotlib import cm


def graph_silhouette(y_km, X):
    cluster_labels = np.unique(y_km)       # y_kmの要素の中で重複を無くす
    n_clusters=cluster_labels.shape[0]     # 配列の長さを返す。つまりここでは n_clustersで指定した3となる
    
    # シルエット係数を計算
    silhouette_vals = silhouette_samples(X,y_km,metric='euclidean')  # サンプルデータ, クラスター番号、ユークリッド距離でシルエット係数計算
    y_ax_lower, y_ax_upper= 0,0
    yticks = []
    
    for i,c in enumerate(cluster_labels):
            c_silhouette_vals = silhouette_vals[y_km==c]      # cluster_labelsには 0,1,2が入っている（enumerateなのでiにも0,1,2が入ってる（たまたま））
            c_silhouette_vals.sort()
            y_ax_upper += len(c_silhouette_vals)              # サンプルの個数をクラスターごとに足し上げてy軸の最大値を決定
            color = cm.jet(float(i)/n_clusters)               # 色の値を作る
            plt.barh(range(y_ax_lower,y_ax_upper),            # 水平の棒グラフのを描画（底辺の範囲を指定）
                             c_silhouette_vals,               # 棒の幅（1サンプルを表す）
                             height=1.0,                      # 棒の高さ
                             edgecolor='none',                # 棒の端の色
                             color=color)                     # 棒の色
            yticks.append((y_ax_lower+y_ax_upper)/2)          # クラスタラベルの表示位置を追加
            y_ax_lower += len(c_silhouette_vals)              # 底辺の値に棒の幅を追加
    
    silhouette_avg = np.mean(silhouette_vals)                 # シルエット係数の平均値
    plt.axvline(silhouette_avg,color="red",linestyle="--")    # 係数の平均値に破線を引く 
    plt.yticks(yticks,cluster_labels + 1)                     # クラスタレベルを表示
    plt.ylabel('Cluster')
    plt.xlabel('silhouette coefficient')
    plt.show()


def main(df_data, max_clusters):
    distortions = []

    for i  in range(2, max_clusters):                # 2~10クラスタまで一気に計算 
        km = KMeans(n_clusters=i,
                    init='k-means++',     # k-means++法によりクラスタ中心を選択
                    n_init=10,
                    max_iter=300,
                    random_state=0)
        km.fit(df_data)                         # クラスタリングの計算を実行
        distortions.append(km.inertia_)   # km.fitするとkm.inertia_が得られる
        y_km = km.fit_predict(df_data)
        graph_silhouette(y_km, df_data)
    
    
    plt.plot(range(2,max_clusters),distortions,marker='o')
    plt.xlabel('Number of clusters')
    plt.ylabel('Distortion')
    plt.show()
    
if __name__=="__main__":
    
    df_data=pd.read_csv("standardized_data.csv", index_col=0, sep=",")
    
    #hyper_parameters
    max_clusters=30
    
    main(df_data, max_clusters)
    
    
   
    
    
    
    
    
