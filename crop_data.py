import os
import numpy as np

from PIL import Image


def crop_image_and_mask(image_path, mask_path, out_image_dir, out_mask_dir, patch_size=256, stride=256, min_crack_pixels=10):
    os.makedirs(out_image_dir, exist_ok=True)
    os.makedirs(out_mask_dir, exist_ok=True)

    img = Image.open(image_path).convert('RGB')
    mask = Image.open(mask_path).convert('L')

    img_name = os.path.splitext(os.path.basename(image_path))[0]
    width, height = img.size


    count = 0

    for top in range(0, height - patch_size + 1, stride):
        for left in range(0, width - patch_size + 1, stride):
            box = (left, top, left + patch_size, top + patch_size)
            img_patch = img.crop(box)
            mask_patch = mask.crop(box)

            if np.array(mask_patch).sum() >= min_crack_pixels:
                img_patch.save(os.path.join(out_image_dir, f"{img_name}_{top}_{left}.jpg"))
                mask_patch.save(os.path.join(out_mask_dir, f"{img_name}_{top}_{left}.png"))
                count+=1
    print(f">>> Đã lưu {count} patch chứa vết nứt từ {img_name}.")

if __name__ == '__main__':
    image_dir = '/home/ngocson/WorkSpace/Datn/project/data/Test/img'
    mask_dir = '/home/ngocson/WorkSpace/Datn/project/data/Val/mask'
    output_img_dir = '/home/ngocson/WorkSpace/Datn/project/data/Test/patches/images'
    output_mask_dir = '/home/ngocson/WorkSpace/Datn/project/data/Val/patches/masks'

    for filename in os.listdir(image_dir):
        if filename.lower().endswith(('.jpg', '.png')):
            img_path = os.path.join(image_dir, filename)
            base_name = os.path.splitext(filename)[0]
            mask_path = os.path.join(mask_dir, base_name + '.png')
            if os.path.exists(mask_path):
                crop_image_and_mask(img_path, mask_path, output_img_dir, output_mask_dir, patch_size=256, stride=256, min_crack_pixels=10)
            else:
                print(f"[!] Không tìm thấy mask cho ảnh: {filename}")