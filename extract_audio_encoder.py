import torch
from models import Wav2Lip

step = '000003000'


def _load(checkpoint_path):
    return torch.load(checkpoint_path)


def extract_audio_encoder(path):
    print("Load checkpoint from: {}".format(path))
    model = Wav2Lip()
    checkpoint = _load(path)
    s = checkpoint["state_dict"]
    new_s = {}
    for k, v in s.items():
        new_s[k.replace('module.', '')] = v
    model.load_state_dict(new_s)

    model = model.to("cuda")
    model.eval()
    torch.save(model.audio_encoder.state_dict(), f'audio_encoder_step{step}.pth')


checkpoints_dir = '/home/ubuntu//Wav2Lip/diginym_w2l_checkpoints'
extract_audio_encoder(f'{checkpoints_dir}/checkpoint_step{step}.pth')

