from src.models.AE import Encoder, Decoder, AE, AE_interp



def build_model(CFG, device: str, loss_type: str = 'Regular'):
    H = CFG['data']['H']
    W = CFG['data']['W']
    latent_dim = CFG['AE']['D']
    N_latent_1 = CFG['AE']['N_latent_1']
    N_latent_2 = CFG['AE']['N_latent_2']
    
    encoder = Encoder(H, W, latent_dim, N_latent_1, N_latent_2)
    decoder = Decoder(H, W, latent_dim, N_latent_1, N_latent_2)

    if loss_type == 'Regular':
        model = AE(encoder, decoder)
    elif loss_type == 'Interp':
        model = AE_interp(encoder, decoder)
    
    return model