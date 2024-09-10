import torch


def restore_checkpoint(ckpt_path, state, device):
    loaded_state = torch.load(ckpt_path, map_location=device)
    if 'pytorch-lightning_version' in loaded_state:
        print('Loading a LiT model...')
        # state['model'].load_state_dict(loaded_state['model'], strict=False)
        raise NotImplementedError('Loading a LiT model is not supported yet')

    else:
        state['optimizer'].load_state_dict(loaded_state['optimizer'])
        state['model'].load_state_dict(loaded_state['model'], strict=False)
        state['ema'].load_state_dict(loaded_state['ema'])
        state['step'] = loaded_state['step']
    return state


def save_checkpoint(ckpt_dir, state):
    saved_state = {
        'optimizer': state['optimizer'].state_dict(),
        'model': state['model'].state_dict(),
        'ema': state['ema'].state_dict() if 'ema' in state else None,  # 'ema' may not exist in 'state
        'step': state['step']
    }
    torch.save(saved_state, ckpt_dir)
