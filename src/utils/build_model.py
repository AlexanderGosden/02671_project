from src.models.AE import Encoder, Decoder, AE

def build_model(model_type: str, CFG, device: str):
    H = CFG['data']['H']
    W = CFG['data']['W']
    latent_dim = CFG['AE']['D']
    N_middle_latent = CFG['AE']['N_middle_latent']
    if model_type == 'AutoEncoder':
        encoder = Encoder(H, W, latent_dim, N_middle_latent)
        decoder = Decoder(H, W, latent_dim, N_middle_latent)
        model = AE(encoder, decoder)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    return model