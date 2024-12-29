import torch

def linear_HSIC(X, Y):
    n = X.shape[0]
    L_X = torch.mm(X, X.T)
    L_Y = torch.mm(Y, Y.T)
    unit = torch.ones([n, n])
    I = torch.eye(n)
    H = (I - unit / n).cuda()
    M = torch.trace(torch.mm(torch.mm(L_X, H), torch.mm(L_Y, H)))
    return M / (n - 1) / (n - 1)


def linear_CKA(X, Y):
    # print(X)
    hsic = linear_HSIC(X, Y)
    var1 = torch.sqrt(linear_HSIC(X, X))
    # print(linear_HSIC(X, X))
    # print('var1:',var1)
    var2 = torch.sqrt(linear_HSIC(Y, Y))
    return hsic / (var1 * var2)