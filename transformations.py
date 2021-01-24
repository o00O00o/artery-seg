import numpy as np
import cv2
import torch


def transforms_for_noise(inputs_u2, std=3e-2):

    gaussian = torch.normal(0, std, inputs_u2.shape)
    inputs_u2_noise = (inputs_u2 + gaussian).contiguous()

    return inputs_u2_noise

def transforms_for_scale(ema_inputs, image_size=64):

    scale_mask = np.random.uniform(low=0.9, high=1.1, size=ema_inputs.shape[0])
    scale_mask = scale_mask * image_size
    scale_mask = [int(item) for item in scale_mask]
    scale_mask = [item - 1 if item % 2 != 0 else item for item in scale_mask]
    half_size = int(image_size / 2)

    ema_outputs = torch.zeros_like(ema_inputs)  # (16, 7, 64, 64)

    for idx in range(ema_inputs.shape[0]):

        img = np.transpose(ema_inputs[idx].cpu().numpy(), (1, 2, 0))  # (64, 64, 7)

        if scale_mask[idx] > image_size:
            pad_width = int((scale_mask[idx] - image_size) / 2)
            img = np.pad(img, ((pad_width, pad_width), (pad_width, pad_width), (0, 0)), 'edge')  # (66, 66, 7)
        else:
            img = img[half_size - int(scale_mask[idx] / 2):half_size + int(scale_mask[idx] / 2), half_size - int(scale_mask[idx] / 2):half_size + int(scale_mask[idx] / 2), :]

        resized_img = cv2.resize(img, (image_size, image_size), interpolation=cv2.INTER_CUBIC)  # (64, 64, 7)

        ema_outputs[idx] = torch.from_numpy(resized_img.transpose((2, 0, 1)))  # (7, 64, 64)

    return ema_outputs.float(), scale_mask

def transforms_back_scale(ema_inputs, scale_mask, image_size=64):
    half_size = int(image_size / 2)
    returned_img = np.zeros((ema_inputs.shape[0], image_size, image_size, ema_inputs.shape[1]))  # (16, 64, 64, 4)
    ema_outputs = torch.zeros_like(ema_inputs)  # (16, 4, 64, 64)

    for idx in range(ema_inputs.shape[0]):
    
        img = np.transpose(ema_inputs[idx].cpu().numpy(), (1, 2, 0))  # (64, 64, 4)
    
        resized_img = cv2.resize(img, (int(scale_mask[idx]), int(scale_mask[idx])), interpolation=cv2.INTER_CUBIC)  # (66, 66, 4)

        if scale_mask[idx] > image_size:
            returned_img[idx] = resized_img[int(scale_mask[idx] / 2) - half_size:int(scale_mask[idx] / 2) + half_size, int(scale_mask[idx] / 2) - half_size:int(scale_mask[idx] / 2) + half_size, :]
        else:
            pad_width = int((image_size - scale_mask[idx]) / 2)
            returned_img[idx] = np.pad(resized_img, ((pad_width, pad_width), (pad_width, pad_width), (0, 0)), 'edge')  # (64, 64, 4)
    
        ema_outputs[idx] = torch.from_numpy(returned_img[idx].transpose((2, 0, 1)))  # (4, 64, 64)

    return ema_outputs  # (16, 4, 64, 64)

def transforms_for_flip(inputs):
    outputs = torch.zeros_like(inputs)
    flip_mask = np.random.randint(0, 2, inputs.shape[0])
    for idx in range(inputs.shape[0]):
        if flip_mask[idx] == 1:
            outputs[idx] = torch.flip(inputs[idx], [1])
        else:
            outputs[idx] = inputs[idx]
    return outputs, flip_mask

def transforms_back_flip(inputs, flip_mask):
    outputs = torch.zeros_like(inputs)
    for idx in range(inputs.shape[0]):
        if flip_mask[idx] == 1:
            outputs[idx] = torch.flip(inputs[idx], [1])
        else:
            outputs[idx] = inputs[idx]
    return outputs

def transforms_for_rot(inputs):
    outputs = torch.zeros_like(inputs)
    rot_mask = np.random.randint(0, 4, inputs.shape[0])

    for idx in range(inputs.shape[0]):
        outputs[idx] = torch.rot90(inputs[idx], int(rot_mask[idx]), dims=[1,2])

    return outputs, rot_mask

def transforms_back_rot(inputs, rot_mask):

    outputs = torch.zeros_like(inputs)
    for idx in range(inputs.shape[0]):
        outputs[idx] = torch.rot90(inputs[idx], int(rot_mask[idx]), dims=[2, 1])
    
    return outputs
