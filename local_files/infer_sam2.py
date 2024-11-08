import os
import shutil

os.chdir("../")

import torch
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sam2.build_sam import build_sam2_video_predictor
from tqdm import tqdm


def show_mask(mask, ax, random_color=False, borders=True):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask = mask.astype(np.uint8)
    mask_image =  mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    if borders:
        import cv2
        contours, _ = cv2.findContours(mask,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        # Try to smooth contours
        contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
        mask_image = cv2.drawContours(mask_image, contours, -1, (1, 1, 1, 0.5), thickness=2)
    ax.imshow(mask_image)

def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))

def show_masks(image, masks, scores, point_coords=None, box_coords=None, input_labels=None, borders=True):
    for i, (mask, score) in enumerate(zip(masks, scores)):
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        show_mask(mask, plt.gca(), borders=borders)
        if point_coords is not None:
            assert input_labels is not None
            show_points(point_coords, input_labels, plt.gca())
        if box_coords is not None:
            # boxes
            show_box(box_coords, plt.gca())
        if len(scores) > 1:
            plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
        plt.axis('off')
        plt.show()


def show_mask_video(mask, ax, obj_id=None, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        cmap = plt.get_cmap("tab10")
        cmap_idx = 0 if obj_id is None else obj_id
        color = np.array([*cmap(cmap_idx)[:3], 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def infer_sam2_img():
    checkpoint = "./checkpoints/sam2_hiera_large.pt"
    model_cfg = "sam2_hiera_l.yaml"
    predictor = SAM2ImagePredictor(build_sam2(model_cfg, checkpoint))

    # img_path = "/home/pxn-lyj/Egolee/programs/UniDexGrasp_liyj/dexgrasp_generation/local_files/data/realsense_data/colors/20_color.jpg"
    # img_path = "/home/pxn-lyj/Egolee/programs/UniDexGrasp_liyj/dexgrasp_generation/local_files/data/render_show_obj000490/realsense_09261548/colors/20_color.jpg"
    # img_path = "/home/pxn-lyj/Egolee/programs/UniDexGrasp_liyj/dexgrasp_generation/local_files/data/render_show_obj000490/realsense_09261855/colors/20_color.jpg"
    img_path = "/home/pxn-lyj/Egolee/data/test/dexgrasp_show_realsense_20241009/colors/20_color.jpg"

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        image = Image.open(img_path)
        image = np.array(image.convert("RGB"))

        predictor.set_image(image)
        # masks, _, _ = predictor.predict(<input_prompts>)
        # masks, _, _ = predictor.predict("cup")

        # input_point = np.array([[500, 375]])
        # input_point = np.array([[371, 331]])
        # input_point = np.array([[364, 167]])
        input_point = np.array([[450, 116]])
        input_label = np.array([1])
        input_box = np.array([427, 89, 470, 158])
        # input_box = None

        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        show_points(input_point, input_label, plt.gca())

        # show_masks(image, masks, scores, point_coords=input_point, input_labels=input_label, borders=True)
        plt.axis('on')
        plt.show()

        print(predictor._features["image_embed"].shape, predictor._features["image_embed"][-1].shape)

        # 输入是点
        # masks, scores, logits = predictor.predict(
        #     point_coords=input_point,
        #     point_labels=input_label,
        #     multimask_output=True,
        # )

        masks, scores, logits = predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            box=input_box[None, :] if input_box is not None else None,
            multimask_output=False,
        )
        # show_masks(image, masks, scores, box_coords=input_box)

        sorted_ind = np.argsort(scores)[::-1]
        # select max score mask
        sorted_ind = [sorted_ind[0]]
        masks = masks[sorted_ind]
        scores = scores[sorted_ind]
        logits = logits[sorted_ind]

        # show_masks(image, masks, scores, point_coords=input_point, input_labels=input_label, borders=True)

        show_masks(image, masks, scores, point_coords=input_point, box_coords=input_box, input_labels=input_label, borders=True)
        np.save("/home/pxn-lyj/Egolee/data/test/dexgrasp_show_realsense_20241009/masks_num_tmp/20_mask.npy", masks[0])

        # input_point = np.array([[500, 375], [1125, 625]])
        # input_label = np.array([1, 1])
        #
        # mask_input = logits[np.argmax(scores), :, :]  # Choose the model's best mask
        #
        # masks, scores, _ = predictor.predict(
        #     point_coords=input_point,
        #     point_labels=input_label,
        #     mask_input=mask_input[None, :, :],
        #     multimask_output=False,
        # )
        #
        # show_masks(image, masks, scores, point_coords=input_point, input_labels=input_label)


def infer_sam2_video():
    import torch
    from sam2.sam2_video_predictor import SAM2VideoPredictor
    # 可以是视频路径或者文件夹路径
    root = "/home/pxn-lyj/Egolee/data/test/dexgrasp_show_realsense_20241012_laser"
    video_dir = os.path.join(root, "colors_num")

    s_mask_dir = os.path.join(root, "masks_num")
    os.makedirs(s_mask_dir, exist_ok=True)

    # input_point = np.array([[318, 330]])   # before 335
    # input_point = np.array([[600, 387]])   # after 335
    # input_point = np.array([[403, 225]])   # after 335
    input_point = np.array([[393, 175]])   # after 335
    frame_names = [
        p for p in os.listdir(video_dir)
        if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
    ]
    frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

    # sam2_checkpoint = "./checkpoints/sam2_hiera_large.pt"
    # model_cfg = "sam2_hiera_l.yaml"

    sam2_checkpoint = "./checkpoints/sam2_hiera_tiny.pt"
    model_cfg = "sam2_hiera_t.yaml"

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # predictor = SAM2VideoPredictor.from_pretrained("facebook/sam2-hiera-large")
    predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        inference_state = predictor.init_state(video_dir, offload_video_to_cpu=True)

        ann_frame_idx = 0  # the frame index we interact with
        ann_obj_id = 1  # give a unique id to each object we interact with (it can be any integers)

        # Let's add a positive click at (x, y) = (210, 350) to get started
        # points = np.array([[210, 350]], dtype=np.float32)
        # for labels, `1` means positive click and `0` means negative click
        labels = np.array([1], np.int32)
        _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=ann_frame_idx,
            obj_id=ann_obj_id,
            points=input_point,
            labels=labels,
        )

        # 显示第一张的mask结果
        plt.figure(figsize=(9, 6))
        plt.title(f"frame {ann_frame_idx}")
        plt.imshow(Image.open(os.path.join(video_dir, frame_names[ann_frame_idx])))
        show_points(input_point, labels, plt.gca())
        show_mask_video((out_mask_logits[0] > 0.0).cpu().numpy(), plt.gca(), obj_id=out_obj_ids[0])

        # 进行video的传播
        video_segments = {}  # video_segments contains the per-frame segmentation results
        for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
            video_segments[out_frame_idx] = {
                out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                for i, out_obj_id in enumerate(out_obj_ids)
            }

        # render the segmentation results every few frames
        # vis_frame_stride = 30
        # plt.close("all")
        # for out_frame_idx in range(0, len(frame_names), vis_frame_stride):
        #     plt.figure(figsize=(6, 4))
        #     plt.title(f"frame {out_frame_idx}")
        #     plt.imshow(Image.open(os.path.join(video_dir, frame_names[out_frame_idx])))
        #     for out_obj_id, out_mask in video_segments[out_frame_idx].items():
        #         show_mask_video(out_mask, plt.gca(), obj_id=out_obj_id)

        # save mask
        object_id = 1
        for i in range(len(frame_names)):
            segments = video_segments[i]
            if object_id in segments:
                mask = segments[object_id][0]
                s_mask_path = os.path.join(s_mask_dir, frame_names[i].split(".")[0] + "_mask.npy")
                np.save(s_mask_path, mask)

        # plt.axis('on')
        # plt.show()


def save_img_names_2_num():
    # sam是视频输入时，需要用数字进行命名
    # root = "/home/pxn-lyj/Egolee/programs/UniDexGrasp_liyj/dexgrasp_generation/local_files/data/render_show_obj000490/realsense_09261855/colors"
    # root = "/home/pxn-lyj/Egolee/programs/UniDexGrasp_liyj/dexgrasp_generation/local_files/data/dexgrasp_show_realsense_20241009/colors"
    # root = "/home/pxn-lyj/Egolee/data/test/dexgrasp_show_realsense_20241011/colors"
    # root = "/home/pxn-lyj/Egolee/data/test/dexgrasp_show_realsense_20241012_laser/colors"
    root = "/home/pxn-lyj/Egolee/data/test/dexgrasp_show_realsense_20241012_no_laser/colors"
    s_root = root + "_num"
    os.makedirs(s_root, exist_ok=True)

    img_names = [name for name in os.listdir(root) if name[-4:] in [".jpg", ".png"]]
    img_names = list(sorted(img_names, key=lambda x: int(x.split(".")[0].split("_")[0])))

    for img_name in tqdm(img_names, desc="img_names"):


        img_path = os.path.join(root, img_name)
        n_img_path = os.path.join(s_root, img_name.split(".")[0].split("_")[0] + img_name[-4:])
        shutil.copy(img_path, n_img_path)
        # if "_color.jpg" not in img_name:
        #     os.remove(img_path)


def show_img_mask():
    mask_path = "/home/pxn-lyj/Egolee/programs/UniDexGrasp_liyj/dexgrasp_generation/local_files/data/render_show_obj000490/realsense_09261855/masks_num/341_mask.npy"
    # mask_path = "/home/pxn-lyj/Egolee/programs/UniDexGrasp_liyj/dexgrasp_generation/local_files/data/render_show_obj000490/realsense_09261855/masks/20_mask.npy"
    mask = np.load(mask_path)

    plt.imshow(mask)
    plt.show()


if __name__ == "__main__":
    print("Start")
    infer_sam2_img()
    # infer_sam2_video()
    # save_img_names_2_num()
    # show_img_mask()
    print("End")
