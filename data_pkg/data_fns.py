import numpy as np
import os
from skimage.io import imread_collection
from tqdm import tqdm
import matplotlib.pyplot as plt
from skimage import color,img_as_float


class Cuboid:

    video_id = (0, 0)
    size_x = 10
    size_y = 10
    n_frames = 5
    frame_size = (128, 128)
    frame_id = int(n_frames / 2)
    x_centroid = 0
    y_centroid = 0
    n_channels = 1
    data = None
    anom_status = False
    anom_score = 0.0
    local_possible = False
    anom_gt = False

    def __init__(self, video_id, size_x, size_y, n_frames, frame_id, x_centroid, y_centroid, n_channels, frame_size,
                 data,anom_gt):

        # Initialize-cuboid-params
        self.video_id = video_id
        self.size_x = size_x
        self.size_y = size_y
        self.n_frames = n_frames
        self.frame_id = frame_id
        self.x_centroid = x_centroid
        self.y_centroid = y_centroid
        self.n_channels = n_channels
        self.frame_size = frame_size
        self.data = data  # shape_data = (size_x,size_y,n_channels,n_frames)
        self.anom_gt = anom_gt

        assert (data.shape == (size_y, size_x, n_channels,n_frames))

    def normed_data(self, data_mean, data_std):

        return (self.data - data_mean) / data_std

    def update_anom_score_local(self, alpha_local, theta_local, dist_local):

        # Local
        if (dist_local > alpha_local):
            self.anom_score += 1
        elif (dist_local <= alpha_local and dist_local >= theta_local):
            self.anom_score += 0.5
        elif (dist_local < theta_local):
            self.anom_score += 0
        else:
            raise ValueError("Invalid distance in local")


    def update_anom_score_global(self, alpha_global, theta_global, dist_global):
        # Global
        if (dist_global > alpha_global):
            self.anom_score += 1
        elif (dist_global <= alpha_global and dist_global >= theta_global):
            self.anom_score += 0.5
        elif (dist_global < theta_global):
            self.anom_score += 0
        else:
            raise ValueError("Invalid distance in global")

    def update_local_possible_true(self):

        self.local_possible = True

    def update_status(self):
        if (self.anom_score >= 1.5):
            self.anom_status = True
        else:
            self.anom_status = False

class MVG:
    mu = None
    sigma = None
    denom_inv = None

    def __init__(self,mu=None,sigma=None):

        self.mu = mu.reshape(mu.shape[0],1)
        self.sigma = sigma
        self.denom_inv = 1/np.sqrt(np.linalg.det(2*np.pi*sigma))
        self.sigma_inv = np.linalg.inv(sigma)

    def prob(self,x):

        mat1 = (x-self.mu).T
        mat2 = (x-self.mu)

        first_product = np.matmul(mat1,self.sigma_inv)
        second_product = np.matmul(first_product,mat2)

        exponent_term = -0.5 * second_product

        exponent = np.exp(exponent_term)[0][0]

        final_value = exponent

        return final_value


def strip_sth(list_to_be_stripped, strip_tag,strip_if_present=True):
    list_relevant = []

    for i in range(0, len(list_to_be_stripped)):

        splitted = list_to_be_stripped[i].split('_')

        if(strip_if_present):
            if (splitted[-1] == strip_tag):
                continue
            else:
                list_relevant.append(list_to_be_stripped[i])
        else:
            if (splitted[-1] != strip_tag):
                continue
            else:
                list_relevant.append(list_to_be_stripped[i])

    return list_relevant


def make_cuboids_from_frames(loc_of_frames, video_id, n_frames, size_x, size_y,test_or_train='Train'):


    assert (n_frames >= 5)  # Frame size have to be greater than 5 for any valuable temporal aspect

    list_images = os.listdir(loc_of_frames)
    list_images.sort()


    list_images_relevant = [os.path.join(loc_of_frames, i) for i in strip_sth(list_images, strip_tag='Store')]

    if(test_or_train=='Test'):
        list_images_gt = os.listdir(loc_of_frames+'_gt')
        list_images_gt.sort()
        list_images_relevant_gt = [os.path.join(loc_of_frames+'_gt', i) for i in strip_sth(list_images_gt, strip_tag='Store')]


    image_collection = imread_collection(list_images_relevant)

    if(test_or_train=='Test'):
        image_collection_gt = imread_collection(list_images_relevant_gt)

    frame_size = image_collection[0].shape

    frame_size_y = frame_size[0]
    frame_size_x = frame_size[1]

    if (len(frame_size) == 3):
        n_channels = frame_size[2]
    else:
        n_channels = 1

    one_side_len = int(n_frames / 2)

    start = one_side_len
    end = len(image_collection) - one_side_len

    num_cubes_x = int(frame_size_x / size_x)
    num_cubes_y = int(frame_size_y / size_y)

    list_cuboids = []
    all_cuboids = []
    all_cuboids_gt = []

    for i in tqdm(range(start, end)):

        local_collection = np.zeros((frame_size_y, frame_size_x, n_channels, n_frames))
        local_collection_gt = np.zeros((frame_size_y, frame_size_x, n_channels, n_frames))
        list_cuboids_local = []

        for j, k in enumerate(range(i - one_side_len, i + one_side_len + 1)):
            img_reshaped = image_collection[k].reshape(frame_size_y, frame_size_x, n_channels)
            local_collection[:, :, :, j] = img_reshaped

            if(test_or_train=='Test'):
                img_reshaped_gt = image_collection_gt[k].reshape(frame_size_y, frame_size_x, n_channels)
                local_collection_gt[:, :, :, j] = img_reshaped_gt

        for j in range(0, num_cubes_y):
            for k in range(0, num_cubes_x):
                start_rows = j * size_y
                end_rows = (j + 1) * size_y

                start_cols = k * size_x
                end_cols = (k + 1) * size_x

                frame_id = i
                y_centroid = (start_rows + end_rows) / 2
                x_centroid = (start_cols + end_cols) / 2

                cuboid_data = local_collection[start_rows:end_rows, start_cols:end_cols, :, :]

                anomaly_gt = False
                if(test_or_train=='Test'):
                    anomaly_gt_sum = np.sum(local_collection_gt[start_rows:end_rows, start_cols:end_cols, :, :])

                    if(anomaly_gt_sum>0):
                        anomaly_gt=True

                assert (cuboid_data.shape == (size_y, size_x, n_channels, n_frames))

                cuboid_obj = Cuboid(video_id, size_x, size_y, n_frames, frame_id, x_centroid, y_centroid, n_channels,
                                    frame_size, cuboid_data,anomaly_gt)

                all_cuboids.append(cuboid_data)
                all_cuboids_gt.append(anomaly_gt)
                list_cuboids_local.append(cuboid_obj)

        array_cuboids = np.array(list_cuboids_local).reshape(num_cubes_y, num_cubes_x)
        list_cuboids.append(array_cuboids)

    return list_cuboids, np.array(all_cuboids),np.array(all_cuboids_gt)


def make_cuboids_of_videos(loc_videos, train_test='Train', size_x=11, size_y=11, n_frames=5,num=-1):

    list_dirs = os.listdir(loc_videos)
    list_dirs.sort()
    list_dirs = strip_sth(list_dirs, strip_tag='Store')
    list_dirs = strip_sth(list_dirs, strip_tag='gt')

    

    if(num!=-1):
        list_dirs=list_dirs[0:num]

    list_cuboids = []

    id_1 = 0
    if (train_test == 'Train'):
        id_1 = 0

    elif (train_test == 'Test'):
        id_1 = 1

    n_channels = 1
    all_cuboids = np.zeros((1, size_y, size_x, n_channels, n_frames))
    all_cuboids_gt = np.array([True])

    for idx, i in enumerate(list_dirs):

        list_cuboids_video, cuboids_video, cuboids_gt = make_cuboids_from_frames(loc_of_frames=os.path.join(loc_videos, i),
                                                                     video_id=(idx, id_1),
                                                                     n_frames=n_frames,
                                                                     size_x=size_x,
                                                                     size_y=size_y,test_or_train=train_test)

        list_cuboids.append(list_cuboids_video)

        all_cuboids = np.vstack((all_cuboids, cuboids_video))
        all_cuboids_gt = np.append(all_cuboids_gt,cuboids_gt)

    return list_cuboids, all_cuboids[1:], all_cuboids_gt[1:]


def make_videos_of_anomalies(list_cuboids_test,path_videos,n_frames,size_x,size_y,threshold,cnn=False):

    list_dirs = os.listdir(path_videos)
    list_dirs.sort()
    list_dirs = strip_sth(list_dirs, strip_tag='Store')
    list_dirs = strip_sth(list_dirs, strip_tag='gt')

    list_videos = []

    for idx, i in enumerate(list_dirs):

        list_videos.append(make_anomaly_video_of_one_video(list_cuboids_test[idx], n_frames, size_x,size_y,threshold,cnn))

    return list_videos


def make_anomaly_video_of_one_video(cuboid_array_list, n_frames, size_x,size_y,threshold,cnn=False):

    img_shape=cuboid_array_list[0][0,0].frame_size
    total_number_of_images = len(cuboid_array_list)+ n_frames - 1

    image_collection_true = np.zeros((total_number_of_images,img_shape[0],img_shape[1]))
    image_collection_anom_detected = np.zeros((total_number_of_images,img_shape[0],img_shape[1]))

    rows = cuboid_array_list[0].shape[0]
    cols = cuboid_array_list[0].shape[1]

    for l in range(0, len(cuboid_array_list)):
        for j in range(0, rows):
            for k in range(0, cols):

                cuboid = cuboid_array_list[l][j, k]

                start_frame = cuboid.frame_id - int((n_frames) / 2)
                end_frame = cuboid.frame_id + int((n_frames) / 2) + 1

                if(cnn):
                    start_rows = cuboid.y_centroid - size_y / 2
                    end_rows = cuboid.y_centroid + size_y / 2
                    start_cols = cuboid.x_centroid - size_x / 2
                    end_cols = cuboid.x_centroid + size_x / 2
                else:
                    start_rows = cuboid.y_centroid - size_y / 2
                    end_rows = cuboid.y_centroid + size_y / 2 + 1
                    start_cols = cuboid.x_centroid - size_x / 2
                    end_cols = cuboid.x_centroid + size_x / 2 + 1


                reshaped_data = np.zeros((n_frames,size_x,size_y))
                nchannels=1

                for z in range(0,n_frames):
                    reshaped_data[z,:,:] = cuboid.data[:,:,0,z]

                image_collection_true[start_frame:end_frame, start_rows:end_rows, start_cols:end_cols] = reshaped_data

                if(cuboid_array_list[l][j,k].anom_score>=threshold):
                    image_collection_anom_detected[start_frame:end_frame, start_rows:end_rows,start_cols:end_cols] = np.ones((n_frames, size_x, size_y))


    return [image_collection_true , image_collection_anom_detected]

def make_anomaly_frames(list_videos,local=False,threshold=1.0):

    list_dirs = []

    for idx, i in enumerate(list_videos):

        directory = os.path.join('Results', 'Video_' + str(idx))

        if not os.path.exists(directory):
            os.makedirs(directory)

        images = i[0]
        mask = i[1]

        for idx_image in range(0, len(images)):

            img = img_as_float(np.uint8(images[idx_image]))
            img_color = np.dstack((img, img, img))
            img_hsv = color.rgb2hsv(img_color)

            color_mask = np.zeros((img.shape[0], img.shape[1], 3))
            color_mask[:, :, 0] = np.uint8(mask[idx_image])
            color_mask_hsv = color.rgb2hsv(color_mask)

            img_hsv[..., 0] = color_mask_hsv[..., 0]
            img_hsv[..., 1] = color_mask_hsv[..., 1] * 0.6

            img_masked = color.hsv2rgb(img_hsv)
            plt.imshow(img_masked)

            if(local):
                plt.savefig(os.path.join(directory, 'image_' + str(idx_image).zfill(3) + '_local_.png'))

            else:
                plt.savefig(os.path.join(directory, 'image_' + str(threshold)+'_'+ str(idx_image).zfill(3) + '.png'))

            plt.close()

        list_dirs.append(os.path.join(os.getcwd(), directory))

    return os.path.join(os.getcwd(),'Results'), list_dirs

def make_videos_of_frames(path_results,list_dirs_of_videos,local=False,threshold=1.5):

    dir = os.path.join(path_results,'Videos')

    if not os.path.exists(dir):
        os.makedirs(dir)

    for idx,i in enumerate(list_dirs_of_videos):

        if(local):
            str_command = 'avconv -r 12 -y -i '+ os.path.join(i,'image_%03d_local_.png') + ' ' + os.path.join(dir,str(idx)+'_local.mp4')
        else:
            str_command = 'avconv -r 12 -y -i ' + os.path.join(i, 'image_'+str(threshold)+'_'+'%03d.png') + ' ' + os.path.join(dir, str(threshold)+'_'+str(idx) + '.mp4')

        os.system(str_command)

    return True


