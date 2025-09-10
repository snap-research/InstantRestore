from natsort import natsorted
from glob import glob

import pyrallis
import torch
import numpy as np
from torchvision import transforms
import sys
sys.path.append(".")
sys.path.append("..")

from face_replace.configs.train_config import TrainConfig
from face_replace.models.face_replace_model import FaceReplaceModel
from face_replace.training.utils.vis_utils import tensor2im
from face_replace.models.attn_processors import SharedAttnProcessor

from torchvision.transforms.v2 import Resize
from face_replace.data.transforms.augmentations import GaussianNoise, JPEGCompress,CustomGaussianBlur

import gradio as gr

"""
TODO:
"""
data_folder = "./gradio_data"
class GradioDemo:
    def __init__(self):
        # Instantiate global variables
        self.model_dict = {
            "Base": "base_ablation_ckpt.pt",
            "AdaIn": "adain_ablation_ckpt.pt",
            "Landmark Attention Model": "lmattn_ablation_ckpt.pt",
            "Final": "final_model_ckpt.pt",
        }

        self.data_dict = {
            "Brie Larson": f"{data_folder}/00001",
            "Martin Freeman": f"{data_folder}/00027",
            "Forest Whitaker": f"{data_folder}/00049",
            "Taraji P. Henson": f"{data_folder}/00052",
            "Rachel McAdams": f"{data_folder}/00062",
            "Chris Pine": f"{data_folder}/00082",
            "Gwyneth Paltrow": f"{data_folder}/00116",
            "Lil Wayne": f"{data_folder}/00168",
            "Blake Lively": f"{data_folder}/00224",
            "Angelina Jolie": f"{data_folder}/00232",
            "Jake Gyllenhaal": f"{data_folder}/00291",
            "Jason Momoa": f"{data_folder}/00435",
            "Bradley Cooper": f"{data_folder}/00479",
            "George Clooney": f"{data_folder}/00621",
            "Michael B. Jordan": f"{data_folder}/00737",
            "Mike Tyson": f"{data_folder}/00749",
            "Natalie Portman": f"{data_folder}/00757",
        }

        self.transform = transforms.Compose([
            transforms.Resize(512, interpolation=transforms.InterpolationMode.LANCZOS),
            transforms.CenterCrop(512),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])
        self.dtype = torch.float16
        torch.backends.cudnn.benchmark = True
        self.device = torch.device('cuda')

        self.select_model('Base')

    # Model selection
    def select_model(self, model_name):
        exp_name = self.model_dict[model_name]
        checkpoint_path = f'/path/to/checkpoints/{exp_name}'
        checkpoint_dict = torch.load(checkpoint_path)
        cfg = pyrallis.decode(TrainConfig, checkpoint_dict['cfg'])
        self.face_replace_model = FaceReplaceModel(cfg=cfg.model, full_cfg=cfg, evaluating=True)
        try:
            out = self.face_replace_model.load_state_dict(checkpoint_dict['state_dict'], strict=True)
        except:
            checkpoint_dict['state_dict'] = {k.replace('.module.', '.') : v for k, v in checkpoint_dict['state_dict'].items()} 
            out = self.face_replace_model.load_state_dict(checkpoint_dict['state_dict'], strict=True)
        self.face_replace_model.eval()
        self.face_replace_model.net.noise_timesteps = [249]
        self.face_replace_model = self.face_replace_model.to(self.device)
        self.need_model_change = False

    def preprocess_inp(self, pil_img):
        input_t = self.transform(pil_img)
        input_t = input_t.unsqueeze(0)
        return input_t

    def preprocess_conds(self, ref_list):
        cond_imgs = [self.transform(ref) for ref in ref_list]
        cond_imgs = torch.stack(cond_imgs, dim=0)
        conds_t = cond_imgs.unsqueeze(0)
        return conds_t

    def predict(self, selected_model, inp_img, ref1, ref2, ref3, ref4):
        if self.need_model_change:
            self.select_model(selected_model)
        inp_t = self.preprocess_inp(inp_img)
        ref_list = [ref1, ref2, ref3, ref4]
        conds_t = self.preprocess_conds(ref_list)
        valid_indices = torch.ones(inp_t.size(0), dtype=int) * 4
        
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                self_attn_processors = [p.float().cpu() for p in self.face_replace_model.net.unet.attn_processors.values() 
                                                    if type(p) == SharedAttnProcessor and p.self_attn_idx is not None]
                for self_attn_processor in self_attn_processors:
                    self_attn_processor.save_self_attentions = True
                
                x_pred, _, _ = self.face_replace_model.net.forward(
                    inp_t.to(self.device, self.dtype),
                    conditioning_images=conds_t.to(self.device, self.dtype),
                    valid_indices=valid_indices,
                )
        
        pred_images = [tensor2im(out, unnorm=True) for out in x_pred][0]
        attn_probs = [p.attention_probs for p in self_attn_processors]
        means = np.zeros(4)
        for idx, attn_prob in enumerate(attn_probs):
            attn_size = attn_prob.shape[2]
            # print("----------------------")
            for cond_img_index in range(0, 4):
                attn_selection = attn_prob[:, :, :, attn_size*(cond_img_index):attn_size*(cond_img_index + 1)]
                # print(cond_img_index)
                # print(attn_selection.mean())
                means[cond_img_index] += attn_selection.mean()
        total = np.sum(means)
        normalized = [round((num / total) * 100, 3) for num in means]
        
        # Adjust the last number to ensure the sum is exactly 100
        normalized[-1] = round(100 - sum(normalized[:-1]), 3)
        return pred_images, normalized

    def update_image(self, celebrity):
        celeb_dir = self.data_dict[celebrity]
        list_urls = [f"{celeb_dir}/degraded.png", f"{celeb_dir}/gt.png"]
        cond_urls = natsorted(glob(f"{celeb_dir}/conditioning/*.png"))[:4]
        return list_urls + cond_urls

    def update_model(self):
        self.need_model_change = True
    
    def degrade_image(self, deg_level, gt_img):
        transform_resize = transforms.Compose([
            transforms.Resize(512, interpolation=transforms.InterpolationMode.LANCZOS),
            transforms.CenterCrop(512),
        ])

        gt_img_resized = transform_resize(gt_img)

        blur_sigma_x = 11.9 * deg_level / 100 + 0.1
        blur_sigma_y = 11.9 * deg_level / 100 + 0.1
        downsample_factor = int(12 * deg_level / 100) + 1
        noise = 15 * deg_level / 100
        quality = 99 - int(39 * deg_level / 100)

        degrade_transforms = transforms.Compose([
            CustomGaussianBlur(41, blur_sigma_x, blur_sigma_y),
            Resize(512//downsample_factor, transforms.InterpolationMode.BILINEAR),
            GaussianNoise(noise),
            JPEGCompress(quality),
            Resize(512, transforms.InterpolationMode.BILINEAR),
        ])

        gt_img_tensor = transforms.ToTensor()(gt_img_resized)
        gt_img_degraded = degrade_transforms(gt_img_tensor)
        gt_img_degraded_pil = transforms.ToPILImage()(gt_img_degraded)
        return gt_img_degraded_pil


    def interface(self):
        with gr.Blocks(css=".centered-markdown {text-align: center;}") as demo:
            gr.Markdown("# Personalized Face Restoration Model Prediction")
            gr.Markdown("""## Instructions
                        - Select one of the models below
                        - To select input data, either:
                            - Select one of the pre-loaded celebrities
                            - Upload your own degraded image and 1-4 reference images
                            - Upload your own clean image and 1-4 reference images, and degrade it
            """)
            with gr.Row():
                selected_model = gr.Radio(list(self.model_dict.keys()), label="Select Model", value="Base")
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("Select a pre-loaded celebrity image set")
                    celebrity_dropdown = gr.Dropdown(list(self.data_dict.keys()), label="Select a Celebrity")
                with gr.Column(scale=3):
                    gr.Markdown("You can also upload a clean image below (and 1-4 reference images) and degrade it using this menu.")
                    deg_slider = gr.Slider(minimum=0, maximum=100, step=1, value=50, label="Choose a degradation level")
                    deg_button = gr.Button("Degrade")
            with gr.Row():
                gr.Markdown("You can also upload your own degraded image and 1-4 reference images")
            with gr.Row():
                with gr.Column(scale=1):
                    input_image = gr.Image(label="Upload Degraded Image", type='pil', image_mode='RGB', interactive=True)
                    gt_image = gr.Image(label="Upload Clean Image", type='pil', image_mode='RGB', interactive=True)
                with gr.Column(scale=1):
                    ref_image1 = gr.Image(label="Upload Reference Image 1", type='pil', image_mode='RGB', interactive=True)
                    ref_image2 = gr.Image(label="Upload Reference Image 2", type='pil', image_mode='RGB', interactive=True)
                with gr.Column(scale=1):
                    ref_image3 = gr.Image(label="Upload Reference Image 3", type='pil', image_mode='RGB', interactive=True)
                    ref_image4 = gr.Image(label="Upload Reference Image 4", type='pil', image_mode='RGB', interactive=True)
                with gr.Column(scale=1):
                    output_image = gr.Image(label="Output Image", interactive=True)
            with gr.Row():
                gr.Markdown("Averaged Attention Weights for each Reference Image")
                attn_weight_ref1 = gr.Number(label="Ref 1")
                attn_weight_ref2 = gr.Number(label="Ref 2")
                attn_weight_ref3 = gr.Number(label="Ref 3")
                attn_weight_ref4 = gr.Number(label="Ref 4")
            
            btn = gr.Button("Predict")

            celebrity_dropdown.change(fn=self.update_image, inputs=celebrity_dropdown, outputs=[input_image, gt_image, ref_image1, ref_image2, ref_image3, ref_image4])
            selected_model.change(fn=self.update_model)

            def run_inference(selected_model, inp_img, ref1, ref2, ref3, ref4):
                result, attn_prob_weights = self.predict(selected_model, inp_img, ref1, ref2, ref3, ref4)
                return result, attn_prob_weights[0], attn_prob_weights[1], attn_prob_weights[2], attn_prob_weights[3]
            
            def run_degrade(deg_slider, gt_image):
                degraded_img = self.degrade_image(deg_slider, gt_image)
                return degraded_img
            
            deg_button.click(run_degrade, inputs=[deg_slider, gt_image], outputs=input_image)
            btn.click(run_inference, inputs=[selected_model, input_image, ref_image1, ref_image2, ref_image3, ref_image4], 
                      outputs=[output_image, attn_weight_ref1, attn_weight_ref2, attn_weight_ref3, attn_weight_ref4] )
        
        demo.launch()

if __name__ == "__main__":
    gradio_demo = GradioDemo()
    gradio_demo.interface()
