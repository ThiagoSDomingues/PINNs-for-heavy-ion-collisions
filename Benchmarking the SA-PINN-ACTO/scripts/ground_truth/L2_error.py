def compute_L2(u_pred, u_true):

    num = np.sqrt(np.mean((u_pred - u_true)**2))
    denom = np.sqrt(np.mean(u_true[0]**2))

    return num / denom
