import numpy as np

    # 对矩阵进行 item 归一化
def item_ratings_scaler(rating, record):
    # m代表用户数量，n代表服务数量
    m, n =rating.shape
    # 保存每个服务的平均响应时间
    rating_mean = np.zeros((n, ))
    # 保存经过正则化后的矩阵
    rating_norm = np.zeros((m,n))
    # 求每个服务的平均值，对每一列求均值
    for i in range(n):
        #第i个服务 对应用户评过分idx 平均得分；
        idx = record[:,i] != 0
        if not np.all(idx == 0):
                rating_mean[i] = np.mean(rating[idx, i])
                rating_norm[idx, i] = rating[idx, i] - rating_mean[i]
        
    return rating_norm, rating_mean

# 对矩阵进行 user 归一化
def user_ratings_normalize(rating, record):
    # m代表用户数量，n代表服务数量
    m, n =rating.shape
    # 保存每个用户的平均响应时间
    rating_mean = np.zeros((m, 1))
    # 保存经过正则化后的矩阵
    rating_norm = np.zeros((m,n))
    # 求每个用户的平均值，对每一列求均值
    for i in range(m):
        # 第i个用户 对应用户评过分idx 平均得分；
        idx = record[i,:] !=0
        if not np.all(idx == 0):
            rating_mean[i] = np.mean(rating[i, idx])
            rating_norm[i, idx] = rating[i, idx] - rating_mean[i]
        
    return rating_norm, rating_mean