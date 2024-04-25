from src.models.AE_interp import Encoder, Decoder, AE_interp

def build_model(model_type: str, CFG, device: str):
    H = CFG['data']['H']
    W = CFG['data']['W']
    latent_dim = CFG['AE']['D']
    N_latent_1 = CFG['AE']['N_latent_1']
    N_latent_2 = CFG['AE']['N_latent_2']
    if model_type == 'AutoEncoder':
        encoder = Encoder(H, W, latent_dim, N_latent_1, N_latent_2)
        decoder = Decoder(H, W, latent_dim, N_latent_1, N_latent_2)
        model = AE_interp(encoder, decoder)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    return model