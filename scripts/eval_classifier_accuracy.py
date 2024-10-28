import os
import gc
import argparse
import datetime
import torch as th
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
from guided_diffusion import sg_util, logger
from guided_diffusion.script_util import (
    NUM_CLASSES,
    create_model_and_diffusion,
    add_dict_to_argparser,
)

def round_to_one_decimal(scale):
    if scale == 0:
        return 0, 0
    
    exponent = int(np.floor(np.log10(scale)))
    coefficient = scale / (10 ** exponent)
    # print("coefficient:", coefficient)
    rounded_coefficient = round(coefficient, 1)
    # print("rounded_coefficient:", rounded_coefficient)
    return rounded_coefficient, exponent

def save_images(results, ref_images, num_rows, num_cols, filename, plot_dir):
    """
    Saves a batch of images and their corresponding samples to the specified directory.
    
    Args:
    results (dict): Dictionary containing the original images and their corresponding samples.
    num_rows (int): Number of rows in the plot.
    num_cols (int): Number of columns in the plot.
    filename (str): Filename for the saved plot.
    plot_dir (str): Directory to save the plots.    

    """

    os.makedirs(plot_dir, exist_ok=True)
    
    # Plot and save images
    fig, axs = plt.subplots(num_rows + 1, num_cols, figsize=(20, 20))
    keys = list(results.keys()) 
    for i in range(num_rows + 1):
        if i == 0:
            data = ref_images
        else:
            data = results[keys[i - 1]]

        data = ((data + 1) * 127.5).clamp(0, 255).to(th.uint8)
        data = data.permute(0, 2, 3, 1).contiguous().cpu().numpy()
        for j in range(num_cols):
            axs[i, j].imshow(data[j])
            axs[i, j].axis('off')
     
    for row in range(num_rows + 1):
        if row == 0:
            axs[row, 0].text(-40, 128, 'Original Images', rotation=90, fontsize=16, va='center')
        elif keys[row - 1] == "x":
            axs[row, 0].text(-40, 128, 'data', rotation=90, fontsize=16, va='center')
        else:
            scale = keys[row - 1].item()
            c, e = round_to_one_decimal(scale) 
            axs[row, 0].text(-20, 128, f's={c}e{e}', rotation=90, fontsize=16, va='center') 
    
    plt.savefig(os.path.join(plot_dir, filename))
    plt.close(fig)


def main():
    args = create_argparser().parse_args()
    logger.log(f'args: {args}')

    if args.log_dir: 
        log_dir_root = args.log_dir
    else: 
        log_dir_root = "logs"
     
    log_dir = os.path.join(
            log_dir_root,
            datetime.datetime.now().strftime("classifier_eval-%Y-%m-%d-%H-%M-%S-%f"),
        ) 
    os.makedirs(log_dir, exist_ok=True) 
    logger.configure(dir=log_dir)

    guide_schedule = th.ones((1000,)).to(sg_util.dev())

    model, diffusion = create_model_and_diffusion(
        image_size=256,
        class_cond=False,
        learn_sigma=True,
        num_channels=256,
        num_res_blocks=2,
        num_head_channels=64,
        attention_resolutions="32,16,8",
        diffusion_steps=1000,
        noise_schedule="linear",
        timestep_respacing=[250],
        use_scale_shift_norm=True,
        resblock_updown=True,
        use_fp16=args.use_fp16,
        use_new_attention_order=True,
        channel_mult="",
        num_heads=4,
        num_heads_upsample=-1,
        dropout=0.0,
        use_kl=False,
        predict_xstart=False,
        rescale_timesteps=False,
        rescale_learned_sigmas=False,
        use_checkpoint=False

    )
    diffusion.guide_schedule = guide_schedule
    model.load_state_dict(
        sg_util.load_state_dict("models/256x256_diffusion_uncond.pt", map_location="cpu")
    )
    model.to(sg_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()

    def model_fn(x, t, y=None, s=None):
        return model(x, t, s=s)


    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    vit_transforms = models.ViT_B_16_Weights.DEFAULT.transforms()

    val_dataset = datasets.ImageNet(root=args.dataset, split='val', transform=transform)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    clf = models.vit_b_16(weights = models.ViT_B_16_Weights.DEFAULT).to(sg_util.dev())
    clf.eval()

    eval_set = [i for i in val_loader][:args.batch_number]

    with th.no_grad():
        logger.log("Measuring base performance...")
        base_correct = 0
        total = 0
        for images, labels in eval_set:
            img = images.to(sg_util.dev())
            img = vit_transforms(img)
            outputs = clf(img).to('cpu')
            _, predicted = th.max(outputs.data, 1)
            total += labels.size(0)
            base_correct += (predicted == labels).sum().item()

            del _, img, outputs, predicted
            th.cuda.empty_cache()
            gc.collect

        accuracy = 100 * base_correct / total
        logger.log(f'Accuracy of the network on the ImageNet validation images: {accuracy:.2f}%')
        
       # results = {}
        scales = [float(i) for i in args.guide_scales.split(",")]
        for scale in scales:
            logger.log(f"Measuring performance at scale {scale}...")
            model_kwargs = {"s" : scale}
            correct = 0
            total = 0
            for images, labels in eval_set:
                img = images.to(sg_util.dev())
                upscale = transforms.Resize(256)
                img = upscale(img)
                
                def cond_fn(x, t, y=None, s=1.0):
                    return (img - x) * s     
                           
                samples, _ = diffusion.p_sample_loop(
                    model_fn,
                    (img.size(0), 3, 256, 256),
                    clip_denoised=args.clip_denoised,
                    model_kwargs=model_kwargs,
                    cond_fn=cond_fn,
                    device=sg_util.dev(),                
                )

                # results[scale] = samples

                # save_images(results=results,num_rows=len(scales),num_cols=args.batch_size,
                # filename=f"{datetime.datetime.now().strftime('Eval_Classifier_Sampling-%Y-%m-%d-%H-%M-%S-%f')}.pdf",plot_dir= args.log_dir)
                
                downscale = transforms.Resize(224)
                samples = downscale(samples)

                logger.log(f"After diffusion before vit_transforms min: {th.min(samples)}, max: {th.max(samples)}")
               
                img = vit_transforms(samples)
                outputs = clf(img).to('cpu')
                _, predicted = th.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                del samples, _, img, outputs, predicted
                th.cuda.empty_cache()
                gc.collect()

            accuracy = 100 * correct / total
            logger.log(f'Accuracy of the network after {scale} strength guiding: {accuracy:.2f}%')

def create_argparser():
    defaults = dict(
        clip_denoised = False,
        guide_scales = "5.0, 6.0, 7.0, 8.0, 9.0, 10.0",
        guide_profile = "constant",
        use_fp16 = True,
        log_dir = "logs",
        batch_size = 32,
        batch_number = 10
    )

    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    parser.add_argument('--dataset')
    return parser

if __name__ == '__main__':
    main()
