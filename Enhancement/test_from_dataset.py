#9.19，16点48分 运行代码：python Enhancement/test_from_dataset.py --chunk_size 1000 --overlap 32

import gc
import os
import argparse
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from glob import glob
from basicsr.models import create_model
from basicsr.utils.options import parse
from skimage.util import img_as_ubyte

def self_ensemble(x, model):
    def forward_transformed(x, hflip, vflip, rotate, model):
        if hflip:
            x = torch.flip(x, (-2,))
        if vflip:
            x = torch.flip(x, (-1,))
        if rotate:
            x = torch.rot90(x, dims=(-2, -1))
        x = model(x)
        if rotate:
            x = torch.rot90(x, dims=(-2, -1), k=3)
        if vflip:
            x = torch.flip(x, (-1,))
        if hflip:
            x = torch.flip(x, (-2,))
        return x
    t = []
    for hflip in [False, True]:
        for vflip in [False, True]:
            for rot in [False, True]:
                t.append(forward_transformed(x, hflip, vflip, rot, model))
    t = torch.stack(t)
    return torch.mean(t, dim=0)

parser = argparse.ArgumentParser(description='Image Enhancement using Retinexformer')
parser.add_argument('--input_dir', default=r'C:\Data_YanWenzhen\Retinexformer\ExDark', type=str, help='Directory of input images')
parser.add_argument('--output_dir', default=r'C:\Data_YanWenzhen\Retinexformer\Enhancement\results', type=str, help='Directory for output results')
parser.add_argument('--opt', type=str, default=r'C:\Data_YanWenzhen\Retinexformer\Options\19lol_v1.yml', help='Path to option YAML file.')
parser.add_argument('--weights', default=r'C:\Data_YanWenzhen\Retinexformer\experiments\(24)19lol_v1\net_g_90000.pth', type=str, help='Path to model weights')
parser.add_argument('--gpus', type=str, default="0", help='GPU devices.')
parser.add_argument('--self_ensemble', action='store_true', help='Use self-ensemble for better results')
parser.add_argument('--chunk_size', type=int, default=1000, help='处理图像的块大小')
parser.add_argument('--overlap', type=int, default=32, help='图像块之间的重叠像素数')
args = parser.parse_args()

# 指定 gpu
gpu_list = ','.join(str(x) for x in args.gpus)
os.environ['CUDA_VISIBLE_DEVICES'] = gpu_list
print('export CUDA_VISIBLE_DEVICES=' + gpu_list)

####### Load yaml #######
opt = parse(args.opt, is_train=False)
opt['dist'] = False

model_restoration = create_model(opt).net_g

# 加载模型
checkpoint = torch.load(args.weights)
model_restoration.load_state_dict(checkpoint['params'])
model_restoration.cuda()
model_restoration = torch.nn.DataParallel(model_restoration)
model_restoration.eval()

# 确保输出目录存在
os.makedirs(args.output_dir, exist_ok=True)

factor = 32
supported_formats = ('.png', '.jpg', '.jpeg', '.JPEG', '.JPG', '.PNG')

input_paths = glob(os.path.join(args.input_dir, '**', '*.*'), recursive=True)
input_paths = [p for p in input_paths if p.lower().endswith(supported_formats)]
print(f'需要处理的图像总数: {len(input_paths)}')


def pad_image(img, factor=32):
    h, w = img.shape[2], img.shape[3]
    H, W = ((h + factor - 1) // factor) * factor, ((w + factor - 1) // factor) * factor
    padh, padw = H - h, W - w
    img_padded = F.pad(img, (0, padw, 0, padh), 'reflect')
    return img_padded, (h, w)


def process_image_in_chunks(model, img, chunk_size=1000, overlap=32):
    _, _, h, w = img.shape
    chunks = []
    restored = torch.zeros_like(img)
    count = torch.zeros_like(img)

    for i in range(0, h, chunk_size - overlap):
        for j in range(0, w, chunk_size - overlap):
            end_i, end_j = min(i + chunk_size, h), min(j + chunk_size, w)
            chunk = img[:, :, i:end_i, j:end_j]
            chunk_padded, _ = pad_image(chunk, factor)
            with torch.no_grad():
                restored_chunk = model(chunk_padded)
            restored_chunk = restored_chunk[:, :, :chunk.shape[2], :chunk.shape[3]]
            restored[:, :, i:end_i, j:end_j] += restored_chunk
            count[:, :, i:i + chunk.shape[2], j:j + chunk.shape[3]] += 1

            # Clear CUDA cache
            torch.cuda.empty_cache()

    restored = restored / count
    return restored


def process_large_image(model, img_path, chunk_size, overlap):
    img = cv2.imread(img_path)
    if img is None:
        print(f"无法读取图像: {img_path}")
        return None

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.float32(img) / 255.
    img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).cuda()

    restored = process_image_in_chunks(model, img, chunk_size, overlap)

    restored = torch.clamp(restored, 0, 1)
    restored = restored.cpu().detach().permute(0, 2, 3, 1).squeeze(0).numpy()
    restored = (restored * 255).astype(np.uint8)
    restored = cv2.cvtColor(restored, cv2.COLOR_RGB2BGR)
    return restored


# 主处理循环
for inp_path in tqdm(input_paths):
    try:
        # 获取图像大小，但不加载整个图像到内存
        img = cv2.imread(inp_path)
        if img is None:
            print(f"无法读取图像: {inp_path}")
            continue

        h, w = img.shape[:2]

        # 总是使用分块处理方法
        restored = process_large_image(model_restoration, inp_path, args.chunk_size, args.overlap)

        if restored is not None:
            output_path = os.path.join(args.output_dir, os.path.relpath(inp_path, args.input_dir))
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            cv2.imwrite(output_path, restored)
        else:
            print(f"处理图像 {inp_path} 失败，无法生成恢复后的图像")

        # 清理 GPU 内存
        torch.cuda.empty_cache()
        gc.collect()

    except Exception as e:
        print(f"处理图像 {inp_path} 时出错: {str(e)}")

print("处理完成。")

# def process_image_in_chunks(model, img, chunk_size=1000, overlap=32):
#     _, _, h, w = img.shape
#     chunks = []
#     for i in range(0, h, chunk_size - overlap):
#         for j in range(0, w, chunk_size - overlap):
#             end_i, end_j = min(i + chunk_size, h), min(j + chunk_size, w)
#             chunk = img[:, :, i:end_i, j:end_j]
#             chunk_padded, _ = pad_image(chunk, factor)
#             with torch.no_grad():
#                 restored_chunk = model(chunk_padded)
#             chunks.append((i, j, restored_chunk[:, :, :chunk.shape[2], :chunk.shape[3]]))
#
#     restored = torch.zeros_like(img)
#     count = torch.zeros_like(img)
#     for i, j, chunk in chunks:
#         restored[:, :, i:i + chunk.shape[2], j:j + chunk.shape[3]] += chunk
#         count[:, :, i:i + chunk.shape[2], j:j + chunk.shape[3]] += 1
#
#     restored = restored / count
#     return restored
#
# def process_large_image(model, img_path, chunk_size, overlap):
#     # 使用OpenCV读取完整图像
#     img = cv2.imread(img_path)
#     if img is None:
#         print(f"无法读取图像: {img_path}")
#         return None
#
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     h, w, _ = img.shape
#     restored = np.zeros((h, w, 3), dtype=np.float32)
#     count = np.zeros((h, w, 1), dtype=np.float32)
#
#     for i in range(0, h, chunk_size - overlap):
#         for j in range(0, w, chunk_size - overlap):
#             end_i, end_j = min(i + chunk_size, h), min(j + chunk_size, w)
#             chunk = img[i:end_i, j:end_j]
#             chunk = np.float32(chunk) / 255.
#             chunk = torch.from_numpy(chunk).permute(2, 0, 1).unsqueeze(0).cuda()
#
#             chunk_padded, _ = pad_image(chunk, factor)
#             with torch.no_grad():
#                 restored_chunk = model(chunk_padded)
#
#             restored_chunk = restored_chunk.squeeze(0).permute(1, 2, 0).cpu().numpy()
#             restored[i:end_i, j:end_j] += restored_chunk[:end_i - i, :end_j - j]
#             count[i:end_i, j:end_j] += 1
#
#     restored = restored / count
#     restored = (restored * 255).clip(0, 255).astype(np.uint8)
#     restored = cv2.cvtColor(restored, cv2.COLOR_RGB2BGR)
#     return restored
#
#
# # 主处理循环
# for inp_path in tqdm(input_paths):
#     try:
#         # 获取图像大小，但不加载整个图像到内存
#         img = cv2.imread(inp_path)
#         if img is None:
#             print(f"无法读取图像: {inp_path}")
#             continue
#
#         h, w = img.shape[:2]
#
#         if max(h, w) > args.chunk_size:
#             restored = process_large_image(model_restoration, inp_path, args.chunk_size, args.overlap)
#         else:
#             img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#             img = np.float32(img) / 255.
#             img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).cuda()
#             img_padded, original_size = pad_image(img, factor)
#
#             if args.self_ensemble:
#                 restored = self_ensemble(img_padded, model_restoration)
#             else:
#                 restored = model_restoration(img_padded)
#
#             restored = restored[:, :, :original_size[0], :original_size[1]]
#             restored = torch.clamp(restored, 0, 1)
#             # 在这里添加 detach() 操作
#             restored = restored.cpu().detach().permute(0, 2, 3, 1).squeeze(0).numpy()
#             restored = (restored * 255).astype(np.uint8)
#             restored = cv2.cvtColor(restored, cv2.COLOR_RGB2BGR)
#
#         if restored is not None:
#             output_path = os.path.join(args.output_dir, os.path.relpath(inp_path, args.input_dir))
#             os.makedirs(os.path.dirname(output_path), exist_ok=True)
#             cv2.imwrite(output_path, restored)
#         else:
#             print(f"处理图像 {inp_path} 失败，无法生成恢复后的图像")
#     except Exception as e:
#         print(f"处理图像 {inp_path} 时出错: {str(e)}")
#
# print("处理完成。")


# # 主处理循环
# for inp_path in tqdm(input_paths):
#     try:
#         # 获取图像大小，但不加载整个图像到内存
#         img = cv2.imread(inp_path)
#         if img is None:
#             print(f"无法读取图像: {inp_path}")
#             continue
#
#         h, w = img.shape[:2]
#
#         if max(h, w) > args.chunk_size:
#             restored = process_large_image(model_restoration, inp_path, args.chunk_size, args.overlap)
#         else:
#             img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#             img = np.float32(img) / 255.
#             img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).cuda()
#             img_padded, original_size = pad_image(img, factor)
#
#             if args.self_ensemble:
#                 restored = self_ensemble(img_padded, model_restoration)
#             else:
#                 restored = model_restoration(img_padded)
#
#             restored = restored[:, :, :original_size[0], :original_size[1]]
#             restored = torch.clamp(restored, 0, 1)
#             # 在这里添加 detach() 操作
#             restored = restored.cpu().detach().permute(0, 2, 3, 1).squeeze(0).numpy()
#             restored = (restored * 255).astype(np.uint8)
#             restored = cv2.cvtColor(restored, cv2.COLOR_RGB2BGR)
#
#         if restored is not None:
#             output_path = os.path.join(args.output_dir, os.path.relpath(inp_path, args.input_dir))
#             os.makedirs(os.path.dirname(output_path), exist_ok=True)
#             cv2.imwrite(output_path, restored)
#         else:
#             print(f"处理图像 {inp_path} 失败，无法生成恢复后的图像")
#     except Exception as e:
#         print(f"处理图像 {inp_path} 时出错: {str(e)}")
#
# print("处理完成。")

# # 主处理循环
# for inp_path in tqdm(input_paths):
#     try:
#         # 获取图像大小
#         img = cv2.imread(inp_path)
#         h, w = img.shape[:2]
#
#         if max(h, w) > args.chunk_size:
#             restored = process_large_image(model_restoration, inp_path, args.chunk_size, args.overlap)
#         else:
#             img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#             img = np.float32(img) / 255.
#             img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).cuda()
#             img_padded, original_size = pad_image(img, factor)
#
#             if args.self_ensemble:
#                 restored = self_ensemble(img_padded, model_restoration)
#             else:
#                 restored = model_restoration(img_padded)
#
#             restored = restored[:, :, :original_size[0], :original_size[1]]
#             restored = torch.clamp(restored, 0, 1).cpu().permute(0, 2, 3, 1).squeeze(0).numpy()
#             restored = (restored * 255).astype(np.uint8)
#             restored = cv2.cvtColor(restored, cv2.COLOR_RGB2BGR)
#
#         if restored is not None:
#             output_path = os.path.join(args.output_dir, os.path.relpath(inp_path, args.input_dir))
#             os.makedirs(os.path.dirname(output_path), exist_ok=True)
#             cv2.imwrite(output_path, restored)
#
#             # 打印处理后图像的尺寸，用于验证
#             print(f"处理完成: {inp_path}, 输出尺寸: {restored.shape[:2]}")
#     except Exception as e:
#         print(f"处理图像 {inp_path} 时出错: {str(e)}")
#
# print("处理完成。")

# def process_large_image(model, img_path, chunk_size, overlap):
#     # 使用OpenCV分块读取大图像
#     img = cv2.imread(img_path)
#     if img is None:
#         print(f"无法读取图像: {img_path}")
#         return None
#
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     h, w, _ = img.shape
#     restored = np.zeros_like(img, dtype=np.float32)
#     count = np.zeros_like(img, dtype=np.float32)
#
#     for i in range(0, h, chunk_size - overlap):
#         for j in range(0, w, chunk_size - overlap):
#             end_i, end_j = min(i + chunk_size, h), min(j + chunk_size, w)
#             chunk = img[i:end_i, j:end_j]
#             chunk = np.float32(chunk) / 255.
#             chunk = torch.from_numpy(chunk).permute(2, 0, 1).unsqueeze(0).cuda()
#
#             chunk_padded, _ = pad_image(chunk, factor)
#             with torch.no_grad():
#                 restored_chunk = model(chunk_padded)
#
#             restored_chunk = restored_chunk.squeeze(0).permute(1, 2, 0).cpu().numpy()
#             restored[i:end_i, j:end_j] += restored_chunk[:end_i - i, :end_j - j]
#             count[i:end_i, j:end_j] += 1
#
#     restored = restored / count
#     restored = (restored * 255).clip(0, 255).astype(np.uint8)
#     restored = cv2.cvtColor(restored, cv2.COLOR_RGB2BGR)
#     return restored
#
#
# # 主处理循环
# for inp_path in tqdm(input_paths):
#     try:
#         # 获取图像大小，但不加载整个图像到内存
#         img = cv2.imread(inp_path)
#         h, w = img.shape[:2]
#
#         if max(h, w) > args.chunk_size:
#             restored = process_large_image(model_restoration, inp_path, args.chunk_size, args.overlap)
#         else:
#             img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#             img = np.float32(img) / 255.
#             img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).cuda()
#             img_padded, original_size = pad_image(img, factor)
#
#             if args.self_ensemble:
#                 restored = self_ensemble(img_padded, model_restoration)
#             else:
#                 restored = model_restoration(img_padded)
#
#             restored = restored[:, :, :original_size[0], :original_size[1]]
#             restored = torch.clamp(restored, 0, 1).cpu().permute(0, 2, 3, 1).squeeze(0).numpy()
#             restored = (restored * 255).astype(np.uint8)
#             restored = cv2.cvtColor(restored, cv2.COLOR_RGB2BGR)
#
#         if restored is not None:
#             output_path = os.path.join(args.output_dir, os.path.relpath(inp_path, args.input_dir))
#             os.makedirs(os.path.dirname(output_path), exist_ok=True)
#             cv2.imwrite(output_path, restored)
#     except Exception as e:
#         print(f"处理图像 {inp_path} 时出错: {str(e)}")
#
# print("处理完成。")





