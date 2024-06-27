from diffusers.image_processor import VaeImageProcessor
import torch


def return_encoder_closure(vae):
  image_processor = VaeImageProcessor(vae_scale_factor=8, do_resize=False, do_normalize=True)
  def encoder(image):
    image = image_processor.preprocess(image)
    image = image.to(device='cuda', dtype=torch.float16)
    with torch.no_grad():
      z = vae.encode(image)
      z = z.latent_dist.mean
      z = torch.multiply(z, torch.tensor(0.18215))
    return z.detach().cpu()
  return encoder

def return_decoder_closure(vae):
  image_processor = VaeImageProcessor(vae_scale_factor=8, do_resize=False, do_normalize=True)
  def decoder(z):
    with torch.no_grad():
      z = z.to('cuda')
      z = torch.divide(z, torch.tensor(0.18215))
      image = vae.decode(z).sample.detach().cpu()

    image = image_processor.postprocess(image)

    return image
  return decoder