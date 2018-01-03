import numpy as np

def ssim(X,Y):

    X = X.flatten()
    Y = Y.flatten()

    mu_x = X.mean()
    mu_y = Y.mean()
    sigma_x = X.std()
    sigma_y = Y.std()

    cv = np.cov(X, Y, rowvar=True)[0,1]

    c1 = 1e-15
    c2 = 1e-15

    n1 = 2*mu_x*mu_y + c1
    n2 = 2*cv + c2

    d1 = mu_x**2 + mu_y**2 + c1
    d2 = sigma_x**2 + sigma_y**2 + c2

    sim = (n1*n2)/(d1*d2)

    return sim


def gen_local_feats(centre_cuboid, surroundings, past_cuboid):

    n_frames = centre_cuboid.shape[-1]

    list_feats = []
    for i in range(0, len(surroundings)):
        consider_cuboid = surroundings[i]
        feat = 0
        for j in range(0, n_frames):
            feat += ssim(consider_cuboid[j, :, :, :], centre_cuboid[j, :, :, :])
        list_feats.append(feat/n_frames)

    feat = 0

    for j in range(0, n_frames):
        feat += ssim(past_cuboid[j, :, :, :], centre_cuboid[j, :, :, :])

    list_feats.append(feat/n_frames)

    for j in range(1, n_frames):
        feat = ssim(centre_cuboid[j-1, :, :, :], centre_cuboid[j, :, :, :])
        list_feats.append(feat)

    return list_feats

def gen_local_feats_temporal(centre_cuboid, surroundings):

    n_frames = centre_cuboid.shape[-1]

    list_feats = []
    for i in range(0, len(surroundings)):
        consider_cuboid = surroundings[i]
        feat = 0
        for j in range(0, n_frames):
            feat += ssim(consider_cuboid[j, :, :, :], centre_cuboid[j, :, :, :])
        list_feats.append(feat/n_frames)


    for j in range(1, n_frames):
        feat = ssim(centre_cuboid[j-1, :, :, :], centre_cuboid[j, :, :, :])
        list_feats.append(feat)

    return list_feats
