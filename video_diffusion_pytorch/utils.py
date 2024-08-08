import re
from functools import wraps
import cv2

import einops
import imageio
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image, ImageFilter
from einops import rearrange, reduce, repeat
from torch import nn
import ot


def resize(im, size, resample=Image.NEAREST):
    """ resize the image to the given number of pixels"""
    if type(im) == np.ndarray:
        im = Image.fromarray(im.astype(np.float32))
    height, width = size
    # pil resize uses opposite convention
    im = im.resize((width, height), resample=resample)
    return im


def norm_img(img, new_max=255.):
    assert img.max() - img.min() >= 0
    img = new_max * (img - img.min()) / (img.max() - img.min())
    return img


def add_additive_noise(img, downsize_factor=8, noise_type="uniform"):
    """ Add Additive random noise to an image """
    if type(img) == Image.Image:
        img = np.array(img)
    img = norm_img(img, new_max=255.)
    r, c = img.shape
    if noise_type == "uniform":
        add_noise = np.random.uniform(0, 100, size=(int(r / downsize_factor), int(c / downsize_factor)))
        sub_noise = np.random.uniform(0, 100, size=(int(r / downsize_factor), int(c / downsize_factor)))
    elif noise_type == "normal":
        add_noise = 10 * np.random.normal(0, 1, size=(int(r / downsize_factor), int(c / downsize_factor)))
        sub_noise = 10 * np.random.normal(0, 1, size=(int(r / downsize_factor), int(c / downsize_factor)))
    else:
        raise ValueError(f"noise type {noise_type} not recognized")
    add_noise = resize(add_noise, (r, c))
    img += add_noise
    sub_noise = resize(sub_noise, (r, c))
    img -= sub_noise
    img = norm_img(img, new_max=255.)
    return img


def gaussian_blur_img(img, blur_kernel_size=5):
    if type(img) == np.ndarray:
        img = Image.fromarray(img)
    img = img.convert("L")
    img = img.filter(ImageFilter.GaussianBlur(blur_kernel_size))
    return np.array(img)


# remap the semantic map to with 0 is the background, 1 is cone, 2 is lv, 3 is myo, 4 is la
final_vals = dict(
    background=0,
    label_background=40,
    lv_blood_pool=70,
    lv_myocardium=190,
    rv_blood_pool=100,
)


def im2mat(img):
    """Converts and image to matrix (one pixel per line)"""
    return img.reshape((img.shape[0] * img.shape[1], img.shape[2]))


def mat2im(X, shape):
    """Converts back a matrix to an image"""
    return X.reshape(shape)


def minmax(img):
    return np.clip(img, 0, 1)


def match_histograms(source_gray, reference_gray):
    rng = np.random.RandomState(42)
    if source_gray.max() > 1:
        source_gray = (source_gray / 255.0).astype(np.float64)
    if reference_gray.max() > 1:
        reference_gray = (reference_gray / 255.0).astype(np.float64)
    original_size = source_gray.shape

    I1 = cv2.resize(source_gray, (256, 256))
    I2 = cv2.resize(reference_gray, (256, 256))

    if len(I1.shape) < 3:
        I1 = np.tile(I1[:, :, np.newaxis], (1, 1, 3))
    if len(I2.shape) < 3:
        I2 = np.tile(I2[:, :, np.newaxis], (1, 1, 3))

    X1 = im2mat(I1)
    X2 = im2mat(I2)

    # training samples
    nb = 500
    idx1 = rng.randint(X1.shape[0], size=(nb,))
    idx2 = rng.randint(X2.shape[0], size=(nb,))

    Xs = X1[idx1, :]
    Xt = X2[idx2, :]

    # SinkhornTransport
    ot_sinkhorn = ot.da.SinkhornTransport(reg_e=1e-1)
    ot_sinkhorn.fit(Xs=Xs, Xt=Xt)
    transp_Xs_sinkhorn = ot_sinkhorn.transform(Xs=X1)
    Image_sinkhorn = minmax(mat2im(transp_Xs_sinkhorn, I1.shape))

    Image_sinkhorn = cv2.resize(Image_sinkhorn, original_size[::-1]).mean(axis=-1)

    return Image_sinkhorn.astype(np.float32) * 255


def add_myo_to_lv(semantic_map):
    dilate_semantic_map = cv2.dilate(semantic_map, np.ones((5, 5), np.uint8), iterations=5)
    semantic_map[np.array(dilate_semantic_map - semantic_map) > 0] = 2

    return semantic_map


def add_cone(semantic_map):
    cone = cv2.imread("./samples/echonet_cone.png", 0)
    cone[cone <= 128] = 0
    cone[cone > 128] = 1
    cone = cv2.resize(cone, semantic_map.shape[::-1], interpolation=cv2.INTER_NEAREST)
    mask = (cone > 0) & (semantic_map > 0)
    cone[mask] = semantic_map[mask]

    return cone


def pseudo_image_from_semantic_map(semantic_map, target_size=(256, 256), is_camus=False):
    """
    Convert a semantic map to a pseudo image, simply by mapping the classes to intensities
    """
    semantic_map = semantic_map.astype(np.float32)
    if not is_camus:
        semantic_map = add_myo_to_lv(semantic_map)
        semantic_map[semantic_map != 0] += 1
        semantic_map = add_cone(semantic_map)

    pseudo_image = np.zeros_like(semantic_map, dtype=np.float32)

    pseudo_image[semantic_map==1] = 0.5

    pseudo_image[semantic_map==2] = 0.0
    pseudo_image[semantic_map==3] = 1
    pseudo_image[semantic_map==4] = 0.2
    # pseudo_image[semantic_map == 0] = 0
    pseudo_image = cv2.resize(pseudo_image, target_size)
    return np.array(pseudo_image* 255).astype(np.float32) 


def check_shape(tensor, pattern, **kwargs):
    return rearrange(tensor, f"{pattern} -> {pattern}", **kwargs)


# do same einops operations on a list of tensors

def _many(fn):
    @wraps(fn)
    def inner(tensors, pattern, **kwargs):
        return (fn(tensor, pattern, **kwargs) for tensor in tensors)

    return inner


# do einops with unflattening of anonymously named dimensions
# (...flattened) ->  ...flattened

def _with_anon_dims(fn):
    @wraps(fn)
    def inner(tensor, pattern, **kwargs):
        regex = r'(\.\.\.[a-zA-Z]+)'
        matches = re.findall(regex, pattern)
        get_anon_dim_name = lambda t: t.lstrip('...')
        dim_prefixes = tuple(map(get_anon_dim_name, set(matches)))

        update_kwargs_dict = dict()

        for prefix in dim_prefixes:
            assert prefix in kwargs, f'dimension list "{prefix}" was not passed in'
            dim_list = kwargs[prefix]
            assert isinstance(dim_list,
                              (list, tuple)), f'dimension list "{prefix}" needs to be a tuple of list of dimensions'
            dim_names = list(map(lambda ind: f'{prefix}{ind}', range(len(dim_list))))
            update_kwargs_dict[prefix] = dict(zip(dim_names, dim_list))

        def sub_with_anonymous_dims(t):
            dim_name_prefix = get_anon_dim_name(t.groups()[0])
            return ' '.join(update_kwargs_dict[dim_name_prefix].keys())

        pattern_new = re.sub(regex, sub_with_anonymous_dims, pattern)

        for prefix, update_dict in update_kwargs_dict.items():
            del kwargs[prefix]
            kwargs.update(update_dict)

        return fn(tensor, pattern_new, **kwargs)

    return inner


# generate all helper functions

rearrange_many = _many(rearrange)
repeat_many = _many(repeat)
reduce_many = _many(reduce)

rearrange_with_anon_dims = _with_anon_dims(rearrange)
repeat_with_anon_dims = _with_anon_dims(repeat)
reduce_with_anon_dims = _with_anon_dims(reduce)

CHANNELS_TO_MODE = {
    1: 'L',
    3: 'RGB',
    4: 'RGBA'
}


def seek_all_images(img, channels=3):
    assert channels in CHANNELS_TO_MODE, f'channels {channels} invalid'
    mode = CHANNELS_TO_MODE[channels]

    i = 0
    while True:
        try:
            img.seek(i)
            yield img.convert(mode)
        except EOFError:
            break
        i += 1


def identity(t, *args, **kwargs):
    return t


def normalize_img(t):
    return t * 2 - 1


def unnormalize_img(t):
    return (t + 1) * 0.5


def video_tensor_from_first_segmap(tensor, cond, path, duration=120, loop=0, optimize=True):
    tensor = einops.rearrange(tensor, 'c t h w -> t h w c')
    images = tensor.detach().cpu().numpy()
    cond = einops.repeat(cond, "h w -> t h w", t=images.shape[0])
    if exists(cond):
        cond = cond.cpu().detach().numpy()[:images.shape[0]]
        cond = cond / cond.max()
        colored_cond = matplotlib.cm.get_cmap('viridis')(cond)[..., :3]
        # where cond not 0, add weighted cond to image with opacity
        images = np.concatenate([images, colored_cond], axis=2)

    images = (images * 255).astype('uint8')
    images = list(map(Image.fromarray, images))
    images[0].save(path, save_all=True, append_images=images[1:], duration=duration, loop=loop, optimize=optimize)


def video_tensor_to_gif(tensor, cond, path, duration=120, loop=0, optimize=True, opacity=0.7, ncol=4):
    """
    Save a video tensor to a gif
    :param tensor: (b, c, t, h, w)
    :param cond: (b, h, w)
    :param path: path to save gif
    """
    # stack batch to ncols
    b, c, t, h, w = tensor.shape
    tensor = tensor[:tensor.shape[0] // ncol * ncol]
    tensor = rearrange(tensor, '(b1 b2) c t h w -> c t (b1 h) (b2 w)', b1=ncol, b2=b // ncol)
    cond = rearrange(cond, '(b1 b2) h w -> (b1 h) (b2 w)', b1=ncol, b2=b // ncol)

    tensor = einops.rearrange(tensor, 'c t h w -> t h w c')
    cond = einops.repeat(cond, "h w -> t h w", t=tensor.shape[0])
    images = tensor.detach().cpu().numpy()
    if exists(cond):
        cond = cond.cpu().detach().numpy()[:images.shape[0]]
        cond = cond / cond.max()
        colored_cond = matplotlib.cm.get_cmap('viridis')(cond)[..., :3]
        # where cond not 0, add weighted cond to image with opacity
        mask = cond > 0
        images[mask] = images[mask] * opacity + colored_cond[mask] * (1 - opacity)
    images = (images * 255).astype('uint8')
    images = list(map(Image.fromarray, images))
    images[0].save(path, save_all=True, append_images=images[1:], duration=duration, loop=loop, optimize=optimize)


def gif_to_tensor(path, channels=3, transform=T.ToTensor()):
    img = Image.open(path)
    tensors = tuple(map(transform, seek_all_images(img, channels=channels)))
    return torch.stack(tensors, dim=1)


def plot_seq(volume, first_segment, save_path):
    """
    Plot the sequence of images and segmentations
    :param volume: (n, c, t, h, w)
    :param first_segment: (n, h, w)
    """

    vis = []
    first_image = volume[0]

    n_line = 40
    grid = np.stack(np.meshgrid(np.linspace(0, 1, n_line), np.linspace(0, 1, n_line)), axis=2)

    batch_size = volume.shape[0]
    org_size = volume.shape[-2:]
    n_image = volume.shape[2]

    imageio.mimsave(save_path, vis, loop=0)


def exists(x):
    return x is not None


def noop(*args, **kwargs):
    pass


def is_odd(n):
    return (n % 2) == 1


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


def cycle(dl):
    while True:
        for data in dl:
            yield data


def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr


def convert_seg2onehot(x, num_classes):
    x = torch.nn.functional.one_hot(x, num_classes=num_classes)
    x = rearrange(x, 'b f h w c -> b c f h w')
    x = x.float()
    return x


def convert_onehot2seg(x):
    x = rearrange(x, 'b c f h w -> b f h w c')
    x = torch.argmax(x, dim=-1)
    return x


def prob_mask_like(shape, prob, device):
    if prob == 1:
        return torch.ones(shape, device=device, dtype=torch.bool)
    elif prob == 0:
        return torch.zeros(shape, device=device, dtype=torch.bool)
    else:
        return torch.zeros(shape, device=device).float().uniform_(0, 1) < prob


def is_list_str(x):
    if not isinstance(x, (list, tuple)):
        return False
    return all([type(el) == str for el in x])


def init_network_weights(m):
    if hasattr(m, 'bias') and m.bias is not None:
        m.bias.data.zero_()
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight)
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
    elif isinstance(m, nn.Linear):
        m.weight.data.normal_(0, 0.01)


def get_jacdet2d(displacement, grid=None, backward=False):
    '''
    Calculate the Jacobian value at each point of the displacement map having
    size of b*2*h*w
    '''
    Dx_x = (displacement[:, 0, :-1, 1:] - displacement[:, 0, :-1, :-1])
    Dx_y = (displacement[:, 0, 1:, :-1] - displacement[:, 0, :-1, :-1])
    Dy_x = (displacement[:, 1, :-1, 1:] - displacement[:, 1, :-1, :-1])
    Dy_y = (displacement[:, 1, 1:, :-1] - displacement[:, 1, :-1, :-1])

    # Dy_x = (displacement[:, 0, :-1, 1:] - displacement[:, 0, :-1, :-1])
    # Dy_y = (displacement[:, 0, 1:, :-1] - displacement[:, 0, :-1, :-1])
    # Dx_x = (displacement[:, 1, :-1, 1:] - displacement[:, 1, :-1, :-1])
    # Dx_y = (displacement[:, 1, 1:, :-1] - displacement[:, 1, :-1, :-1])

    # Normal grid
    if backward:
        D1 = (1 - Dx_x) * (1 - Dy_y)
    else:
        D1 = (1 + Dx_x) * (1 + Dy_y)
    # D1 = (Dx_x) * (Dy_y)
    D2 = Dx_y * Dy_x
    jacdet = D1 - D2

    # # tanh grid
    # grid_x = grid[:, 0, :-1, :-1]
    # grid_y = grid[:, 1, :-1, :-1]
    # coef = 1 - torch.tanh(torch.atanh(grid) + displacement)**2
    # coef_x = coef[:, 0, :-1, :-1]
    # coef_y = coef[:, 1, :-1, :-1]
    # D1 = (1 / (1 - grid_x**2) + Dx_x) * (1 / (1 - grid_y**2) + Dy_y)
    # D2 = Dx_y * Dy_x
    # jacdet = coef_x * coef_y * (D1 - D2)

    return jacdet


def jacdet_loss(vf, grid=None, backward=False):
    '''
    Penalizing locations where Jacobian has negative determinants
    Add to final loss
    '''
    # jacdet = get_jacdet2d(vf, grid, backward)
    # ans = 1 / 2 * (torch.abs(jacdet) - jacdet).mean(axis=[1, 2]).sum()
    # return ans

    jacdet = get_jacdet2d(vf, grid, backward)
    ans = 1 / 2 * (torch.abs(jacdet[jacdet > 1]).mean() + torch.abs(jacdet[jacdet < 0]).mean())
    if torch.isnan(ans):
        return 0
    return ans


def constrained_jacdet_loss(vf, grid=None, backward=False):
    '''
    regularization to ensure jac only range from 0-1
    '''
    jacdet = get_jacdet2d(vf, grid, backward)
    ans = 1 / 2 * (torch.abs(jacdet) - jacdet).mean(axis=[1, 2]).sum()
    return ans


def outgrid_loss(vf, grid, backward=False, size=32):  # 32-1):
    '''
    Penalizing locations where Jacobian has negative determinants
    Add to final loss
    '''
    if backward:
        pos = grid - vf - (size - 1)
        neg = grid - vf
    else:
        pos = grid + vf - (size - 1)
        neg = grid + vf

    # penalize > size
    ans_p = 1 / 2 * (torch.abs(pos) + pos).mean(axis=[1, 2]).sum()
    # penalize < 0
    ans_n = 1 / 2 * (torch.abs(neg) - neg).mean(axis=[1, 2]).sum()
    ans = ans_n + ans_p

    return ans


if __name__ == '__main__':
    import matplotlib

    segmap = cv2.imread("samples/camus/segmap.png", 0)
    colored_segmap = matplotlib.cm.get_cmap('viridis')(segmap / segmap.max())[..., :3]
    pseudo_image = pseudo_image_from_semantic_map(segmap, target_size=(256, 256), is_camus=True)
    # cv2.imshow("pseudo_image", cv2.resize(np.array(pseudo_image * 255).astype('uint8'), (256, 256)))
    # cv2.imshow("colored_segmap", cv2.resize(np.array(colored_segmap * 255).astype('uint8'), (256, 256)))
    # cv2.waitKey(0)
    # # convert bgr to rgb
    colored_segmap = colored_segmap[..., ::-1]
    # # nose sampling with mean is pseudo image and std is 0.1
    # noisy_pseudo_image = pseudo_image * 2 - 1
    # noisy_pseudo_image = np.random.normal(noisy_pseudo_image, 0.1)
    # noisy_pseudo_image = np.clip(noisy_pseudo_image, -1, 1)
    # noisy_pseudo_image = (noisy_pseudo_image + 1) * 0.5
    #
    cv2.imwrite("pseudo_image.png", cv2.resize(np.array(pseudo_image * 255).astype('uint8'), (256, 256)))
    cv2.imwrite("colored_segmap.png", cv2.resize(np.array(colored_segmap * 255).astype('uint8'), (256, 256)))
    # cv2.imwrite("noisy_pseudo_image.png", cv2.resize(np.array(noisy_pseudo_image * 255).astype('uint8'), (256, 256)))
