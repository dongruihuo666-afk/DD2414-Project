#!/usr/bin/env python3
"""Build a presentation-friendly camera vs. camera+radar comparison board."""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


PANEL_SIZE = 240
MARGIN = 24
BACKGROUND = (245, 247, 250)
TEXT = (25, 30, 40)


def font(size, bold=False):
    candidates = (
        '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'
        if bold else '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
        '/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf'
        if bold else '/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf',
    )
    for candidate in candidates:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


def probability_image(values):
    values = np.clip(values, 0, 1)
    rgb = np.zeros((*values.shape, 3), dtype=np.uint8)
    rgb[..., 0] = (255 * values).astype(np.uint8)
    rgb[..., 1] = (90 * np.sqrt(values)).astype(np.uint8)
    return Image.fromarray(rgb).resize(
        (PANEL_SIZE, PANEL_SIZE), Image.Resampling.NEAREST
    )


def binary_image(values, color):
    mask = values > 0.5
    rgb = np.zeros((*values.shape, 3), dtype=np.uint8)
    rgb[mask] = color
    return Image.fromarray(rgb).resize(
        (PANEL_SIZE, PANEL_SIZE), Image.Resampling.NEAREST
    )


def radar_prediction_image(prediction, radar):
    rgb = np.asarray(probability_image(prediction)).copy()
    radar_up = np.asarray(
        Image.fromarray((radar > 0).astype(np.uint8) * 255).resize(
            (PANEL_SIZE, PANEL_SIZE), Image.Resampling.NEAREST
        )
    ) > 0
    rgb[radar_up] = (0, 235, 255)
    return Image.fromarray(rgb)


def metric_text(data):
    return (
        f"IoU {float(data['iou']):.3f}   "
        f"forward {float(data['forward_seconds']):.3f}s   "
        f"VRAM {float(data['peak_cuda_memory_gib']):.2f} GiB"
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--camera', type=Path, required=True)
    parser.add_argument('--radar', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    camera = np.load(args.camera)
    radar = np.load(args.radar)
    if not np.array_equal(camera['ground_truth'], radar['ground_truth']):
        raise ValueError('camera and radar snapshots are not from the same sample')

    width = 5 * PANEL_SIZE + 6 * MARGIN
    camera_height = 180
    height = 90 + 2 * camera_height + 80 + PANEL_SIZE + 105
    canvas = Image.new('RGB', (width, height), BACKGROUND)
    draw = ImageDraw.Draw(canvas)
    title_font = font(30, bold=True)
    heading_font = font(20, bold=True)
    body_font = font(16)
    small_font = font(14)

    draw.text(
        (MARGIN, 20), 'Simple-BEV on nuScenes mini: what camera and radar add',
        font=title_font, fill=TEXT,
    )
    draw.text(
        (MARGIN, 60),
        'One sample | 6 cameras | 112x192 input | 200x200 BEV | official checkpoints',
        font=body_font, fill=(70, 78, 92),
    )

    camera_width = (width - 4 * MARGIN) // 3
    camera_names = (
        'front (BEV reference)', 'front-left', 'front-right',
        'back-left', 'back', 'back-right',
    )
    for index, image_array in enumerate(camera['rgb']):
        image = Image.fromarray(image_array).resize(
            (camera_width, camera_height), Image.Resampling.BILINEAR
        )
        col = index % 3
        row = index // 3
        x = MARGIN + col * (camera_width + MARGIN)
        y = 90 + row * camera_height
        canvas.paste(image, (x, y))
        label_box = draw.textbbox((0, 0), camera_names[index], font=small_font)
        label_width = label_box[2] - label_box[0] + 14
        draw.rectangle((x, y, x + label_width, y + 25), fill=(0, 0, 0))
        draw.text(
            (x + 7, y + 4), camera_names[index], font=small_font, fill='white'
        )

    panel_y = 90 + 2 * camera_height + 58
    panels = (
        ('Ground truth', binary_image(camera['ground_truth'], (0, 255, 80))),
        ('Radar returns', binary_image(radar['radar_bev'], (0, 235, 255))),
        ('Camera-only', probability_image(camera['prediction'])),
        ('Camera + radar', probability_image(radar['prediction'])),
        ('Radar overlay', radar_prediction_image(radar['prediction'], radar['radar_bev'])),
    )
    for index, (label, panel) in enumerate(panels):
        x = MARGIN + index * (PANEL_SIZE + MARGIN)
        draw.text((x, panel_y - 31), label, font=heading_font, fill=TEXT)
        canvas.paste(panel, (x, panel_y))

    metric_y = panel_y + PANEL_SIZE + 18
    draw.text((MARGIN, metric_y), 'Camera-only: ' + metric_text(camera), font=body_font, fill=TEXT)
    draw.text((MARGIN, metric_y + 25), 'Camera+radar: ' + metric_text(radar), font=body_font, fill=TEXT)
    draw.text(
        (MARGIN, metric_y + 54),
        'Green = labeled vehicles | red/orange = predicted probability | cyan = radar-supported cells',
        font=small_font, fill=(70, 78, 92),
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.output)
    print(args.output)


if __name__ == '__main__':
    main()
