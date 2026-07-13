from __future__ import annotations

import shutil
from collections import Counter
from math import inf
from pathlib import Path

import cv2
import numpy as np
import pytesseract
from PIL import Image


PADDING_COLORS: list[tuple[int, int, int, int]] = [
    (255, 0, 255, 255),
    (0, 255, 255, 255),
    (255, 255, 0, 255),
    (1, 2, 3, 255),
    (255, 1, 2, 255),
]


def list_images(directory: Path) -> list[Path]:
    extensions = {".png", ".jpg", ".jpeg"}
    return sorted(
        p for p in directory.iterdir()
        if p.is_file() and p.suffix.lower() in extensions
    )

def load_rgba(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("RGBA"))

def prepare_work_dir(in_dir: Path, out_dir: Path) -> None:
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    for path in list_images(in_dir):
        shutil.copy2(path, out_dir / path.name)

def get_img_map(paths: list[Path]) -> dict[Path, np.ndarray]:
    ret: dict[Path, np.ndarray] = {}
    for path in paths:
        try:
            ret[path] = load_rgba(path)
        except OSError:
            continue
    return ret

def save_rgba(path: Path, pixels: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(pixels.astype(np.uint8)).save(path)

def ensure_rgba(pixels: np.ndarray) -> np.ndarray:
    if pixels.ndim == 2:
        gray = pixels[:, :, np.newaxis]
        pixels = np.repeat(gray, 3, axis=2)
    if pixels.shape[2] == 3:
        alpha = np.full(pixels.shape[:2] + (1,), 255, dtype=pixels.dtype)
        pixels = np.concatenate([pixels, alpha], axis=2)
    return pixels.astype(np.uint8)

def to_match_channels(pixels: np.ndarray) -> np.ndarray:
    rgba = ensure_rgba(pixels)
    return cv2.cvtColor(rgba, cv2.COLOR_RGBA2RGB)

def find_unused_color(*images: np.ndarray) -> tuple[int, int, int, int]:
    candidates = [
        (255, 0, 255, 255),
        (0, 255, 255, 255),
        (255, 255, 0, 255),
        (1, 2, 3, 255),
        (255, 1, 2, 255),
    ]
    for color in candidates:
        color_arr = np.array(color, dtype=np.uint8)
        if all(not np.any(np.all(ensure_rgba(img) == color_arr, axis=2)) for img in images):
            return color
    raise RuntimeError("Nie znaleziono koloru tła, którego nie ma w obrazach")

def paste_rgba(canvas: np.ndarray, overlay: np.ndarray, x: int, y: int) -> None:
    overlay = ensure_rgba(overlay)
    h, w = overlay.shape[:2]
    canvas_h, canvas_w = canvas.shape[:2]

    x0 = max(0, x)
    y0 = max(0, y)
    x1 = min(canvas_w, x + w)
    y1 = min(canvas_h, y + h)

    if x0 >= x1 or y0 >= y1:
        return

    overlay_x0 = x0 - x
    overlay_y0 = y0 - y
    overlay_x1 = overlay_x0 + (x1 - x0)
    overlay_y1 = overlay_y0 + (y1 - y0)

    region = canvas[y0:y1, x0:x1]
    patch = overlay[overlay_y0:overlay_y1, overlay_x0:overlay_x1]
    alpha = patch[:, :, 3:4].astype(np.float32) / 255.0
    region[:] = (patch.astype(np.float32) * alpha + region.astype(np.float32) * (1.0 - alpha)).astype(
        np.uint8
    )

def crop_background(pixels: np.ndarray, bg_color: tuple[int, int, int, int], tolerance: int = 8) -> np.ndarray:
    rgba = ensure_rgba(pixels)
    bg = np.array(bg_color[:4], dtype=np.int16)
    diff = np.max(np.abs(rgba.astype(np.int16) - bg), axis=2)
    mask = diff > tolerance
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)

    if not rows.any() or not cols.any():
        return rgba

    y1, y2 = np.where(rows)[0][[0, -1]]
    x1, x2 = np.where(cols)[0][[0, -1]]
    return rgba[y1 : y2 + 1, x1 : x2 + 1]


def quantize_rgba(pixel: np.ndarray, step: int = 8) -> tuple[int, int, int, int]:
    return tuple((int(pixel[i]) // step) * step for i in range(4))


def pixel_matches_known_border(pixel: np.ndarray, tolerance: int = 20) -> bool:
    px = pixel.astype(np.int16)
    for color in PADDING_COLORS:
        if np.max(np.abs(px - np.array(color, dtype=np.int16))) <= tolerance:
            return True

    r, g, b, a = (int(px[0]), int(px[1]), int(px[2]), int(px[3]))
    if a < 200:
        return False
    if r > 200 and b > 200 and g < 80:
        return True
    if g > 200 and b > 200 and r < 80:
        return True
    if r > 200 and g > 200 and b < 80:
        return True
    return False


def build_border_color_bins(rgba: np.ndarray, ring: int = 50) -> set[tuple[int, int, int, int]]:
    h, w = rgba.shape[:2]
    ring = min(ring, h // 4, w // 4)
    center = rgba[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4].reshape(-1, 4)
    center_bins = Counter(quantize_rgba(px) for px in center)
    center_total = max(len(center), 1)

    outer_pixels = np.vstack(
        [
            rgba[:ring, :, :].reshape(-1, 4),
            rgba[h - ring :, :, :].reshape(-1, 4),
            rgba[:, :ring, :].reshape(-1, 4),
            rgba[:, w - ring :, :].reshape(-1, 4),
        ]
    )
    outer_bins = Counter(quantize_rgba(px) for px in outer_pixels)
    outer_total = max(len(outer_pixels), 1)

    border_bins: set[tuple[int, int, int, int]] = set()
    for bin_color, outer_count in outer_bins.items():
        if outer_count / outer_total < 0.004:
            continue
        center_count = center_bins.get(bin_color, 0)
        if center_count / center_total < 0.01:
            border_bins.add(bin_color)

    return border_bins


def pixel_is_border(
    pixel: np.ndarray,
    border_bins: set[tuple[int, int, int, int]],
    tolerance: int = 10,
) -> bool:
    if pixel_matches_known_border(pixel):
        return True

    quantized = quantize_rgba(pixel)
    if quantized in border_bins:
        return True

    px = pixel.astype(np.int16)
    for bin_color in border_bins:
        if np.max(np.abs(px - np.array(bin_color, dtype=np.int16))) <= tolerance:
            return True
    return False


def line_border_ratio(line: np.ndarray, border_bins: set[tuple[int, int, int, int]]) -> float:
    flat = line.reshape(-1, 4)
    if flat.size == 0:
        return 0.0
    border_pixels = sum(pixel_is_border(px, border_bins) for px in flat)
    return border_pixels / len(flat)


def detect_border_insets(
    pixels: np.ndarray,
    max_scan: int = 120,
    threshold: float = 0.75,
) -> tuple[int, int, int, int]:
    rgba = ensure_rgba(pixels)
    h, w = rgba.shape[:2]
    border_bins = build_border_color_bins(rgba)

    top = 0
    while top < min(max_scan, h // 2) and line_border_ratio(rgba[top, :, :], border_bins) >= threshold:
        top += 1

    bottom = 0
    while bottom < min(max_scan, h // 2) and line_border_ratio(rgba[h - 1 - bottom, :, :], border_bins) >= threshold:
        bottom += 1

    left = 0
    while left < min(max_scan, w // 2) and line_border_ratio(rgba[:, left, :], border_bins) >= threshold:
        left += 1

    right = 0
    while right < min(max_scan, w // 2) and line_border_ratio(rgba[:, w - 1 - right, :], border_bins) >= threshold:
        right += 1

    return top, bottom, left, right


def crop_detected_border(
    pixels: np.ndarray,
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    rgba = ensure_rgba(pixels)
    h, w = rgba.shape[:2]
    top, bottom, left, right = detect_border_insets(rgba)

    if top + bottom >= h or left + right >= w:
        return rgba, (top, bottom, left, right)

    return rgba[top : h - bottom, left : w - right], (top, bottom, left, right)


def sample_column_background(
    rgba: np.ndarray,
    x: int,
    footer_top: int,
) -> np.ndarray:
    y_end = max(0, footer_top - 10)
    y_start = max(0, footer_top - 50)

    if y_start >= y_end:
        column = rgba[:footer_top, x, :].reshape(-1, 4).astype(np.int16)
    else:
        column = rgba[y_start:y_end, x, :].reshape(-1, 4).astype(np.int16)

    if column.size == 0:
        column = rgba[footer_top:, x, :].reshape(-1, 4).astype(np.int16)

    return np.median(column, axis=0).astype(np.uint8)


def detect_footer_top(rgba: np.ndarray, footer_height: int = 130) -> int:
    h = rgba.shape[0]
    footer_top = max(0, h - footer_height)

    for y in range(footer_top - 1, max(0, footer_top - 35), -1):
        row = rgba[y, :, :3].astype(np.float32)
        luminance = 0.299 * row[:, 0] + 0.587 * row[:, 1] + 0.114 * row[:, 2]
        row_median = np.median(luminance)

        if np.mean(luminance > row_median + 18) > 0.004:
            footer_top = y
        else:
            break

    return footer_top


def remove_footer_watermarks(pixels: np.ndarray) -> np.ndarray:
    rgba = ensure_rgba(pixels).copy()
    h, w = rgba.shape[:2]
    footer_top = detect_footer_top(rgba)

    if footer_top >= h:
        return rgba

    replaced = 0
    for x in range(w):
        background = sample_column_background(rgba, x, footer_top)
        background_lum = 0.299 * background[0] + 0.587 * background[1] + 0.114 * background[2]

        for y in range(footer_top, h):
            pixel = rgba[y, x]
            pixel_lum = 0.299 * pixel[0] + 0.587 * pixel[1] + 0.114 * pixel[2]
            color_distance = np.max(np.abs(pixel.astype(np.int16) - background.astype(np.int16)))

            if pixel_lum > background_lum + 5 or color_distance > 10:
                rgba[y, x] = background
                replaced += 1

    print(f"Usunięto watermarki w stopce: {replaced} pikseli (footer_top={footer_top})")
    return rgba


def crop_to_content(image_path: Path, output_path: Path) -> bool:
    try:
        pixels = load_rgba(image_path)
    except OSError:
        print(f"Nie udało się wczytać obrazka: {image_path}")
        return False

    cropped, (top, bottom, left, right) = crop_detected_border(pixels)
    cropped = remove_footer_watermarks(cropped)

    save_rgba(output_path, cropped)
    print(
        f"Przycięto {image_path.name}: {pixels.shape[1]}x{pixels.shape[0]} "
        f"-> {cropped.shape[1]}x{cropped.shape[0]} "
        f"(inset: top={top}, bottom={bottom}, left={left}, right={right})"
    )
    return True


def merge_images(
    first_pixels: np.ndarray,
    second_pixels: np.ndarray,
    x_shift: int,
    y_shift: int,
) -> np.ndarray:
    first = ensure_rgba(first_pixels)
    second = ensure_rgba(second_pixels)

    h1, w1 = first.shape[:2]
    h2, w2 = second.shape[:2]

    side_margin = abs(w1 - w2)
    bottom_margin = h1 + h2

    min_x = min(0, x_shift)
    min_y = min(0, y_shift)
    max_x = max(w1, x_shift + w2)
    max_y = max(h1, y_shift + h2)

    content_w = max_x - min_x
    content_h = max_y - min_y

    canvas_w = content_w + 2 * side_margin
    canvas_h = content_h + bottom_margin

    bg_color = find_unused_color(first, second)
    canvas = np.full((canvas_h, canvas_w, 4), bg_color, dtype=np.uint8)

    first_x = side_margin - min_x
    first_y = -min_y
    second_x = first_x + x_shift
    second_y = first_y + y_shift

    paste_rgba(canvas, first, first_x, first_y)
    paste_rgba(canvas, second, second_x, second_y)

    return crop_background(canvas, bg_color)

class MatchOutputInfo:
    def __init__(self, score: float, x_shift: int, y_shift: int, height: int):
        self.score = score
        self.x_shift = x_shift
        self.y_shift = y_shift
        self.height = height

    def get_values_from(self, other: MatchOutputInfo) -> None:
        self.score = other.score
        self.x_shift = other.x_shift
        self.y_shift = other.y_shift
        self.height = other.height

def single_match_similarity_score(
    first: np.ndarray,
    second: np.ndarray,
    pattern_height: int,
    acceptable_margines: int,
    first_name: str,
    second_name: str,
) -> MatchOutputInfo:
    first_rgb = to_match_channels(first)
    second_rgb = to_match_channels(second)

    if pattern_height <= 0:
        return MatchOutputInfo(inf, 0, 0, 0)

    if acceptable_margines * 2 >= second_rgb.shape[1]:
        return MatchOutputInfo(inf, 0, 0, 0)

    wzorzec = second_rgb[0:pattern_height, acceptable_margines:-acceptable_margines]

    if wzorzec.size == 0 or wzorzec.shape[0] > first_rgb.shape[0] or wzorzec.shape[1] > first_rgb.shape[1]:
        return MatchOutputInfo(inf, 0, 0, 0)

    try:
        wynik_dopasowania = cv2.matchTemplate(first_rgb, wzorzec, cv2.TM_SQDIFF_NORMED)
        min_wartosc, _, min_kordy, _ = cv2.minMaxLoc(wynik_dopasowania)
        znalezione_x, znalezione_y = min_kordy

        przesuniecie_x = znalezione_x - acceptable_margines
        przesuniecie_y = znalezione_y

        print(
            f"comparing: {first_name} vs {second_name} | "
            f"score: {min_wartosc} | y_shift: {przesuniecie_y} | "
            f"x_shift: {przesuniecie_x} | pattern_height: {pattern_height}"
        )
        return MatchOutputInfo(min_wartosc, przesuniecie_x, przesuniecie_y, pattern_height)
    except cv2.error:
        return MatchOutputInfo(inf, 0, 0, 0)

def find_best_match(
    first_pixels: np.ndarray,
    second_pixels: np.ndarray,
    acceptable_margines: int,
    first_name: str,
    second_name: str,
) -> MatchOutputInfo:
    pattern_height = 10
    current = single_match_similarity_score(
        first_pixels, second_pixels, pattern_height, acceptable_margines, first_name, second_name
    )

    second_height = second_pixels.shape[0]
    max_pattern_height = second_height - 1

    while pattern_height < max_pattern_height:
        pattern_height += 1
        candidate = single_match_similarity_score(
            first_pixels, second_pixels, pattern_height, acceptable_margines, first_name, second_name
        )

        if candidate.score > 0.001:
            break

        if candidate.score <= current.score:
            current.get_values_from(candidate)

    print(f"comparing: {first_name} vs {second_name} | biggest pattern height: {current.height}")
    return current

def remove_image_entry(inputs: list[tuple[Path, np.ndarray]], path: Path) -> None:
    path.unlink(missing_ok=True)
    inputs[:] = [(p, pixels) for p, pixels in inputs if p != path]

def look_for_first_match(
    inputs: list[tuple[Path, np.ndarray]],
    work_dir: Path,
) -> bool:
    acceptable_margines = 60
    inputs_count = len(inputs)

    for first_idx in range(inputs_count):
        for second_idx in range(inputs_count):
            if first_idx == second_idx:
                continue

            first_path, first_pixels = inputs[first_idx]
            second_path, second_pixels = inputs[second_idx]

            print(f"first.path: {first_path} first.pixels.shape: {first_pixels.shape}")
            print(f"second.path: {second_path} second.pixels.shape: {second_pixels.shape}\n")

            match_info = find_best_match(
                first_pixels,
                second_pixels,
                acceptable_margines,
                first_path.name,
                second_path.name,
            )

            if match_info.score > 0.001:
                continue

            merged_pixels = merge_images(
                first_pixels,
                second_pixels,
                match_info.x_shift,
                match_info.y_shift,
            )

            merged_name = f"merged_{first_path.stem}_{second_path.stem}.png"
            merged_path = work_dir / merged_name
            save_rgba(merged_path, merged_pixels)

            print(f"Zapisano zmergowany obraz: {merged_path}")

            remove_image_entry(inputs, first_path)
            remove_image_entry(inputs, second_path)
            inputs.append((merged_path, merged_pixels))

            return True

    return False


def list_merged_images(directory: Path) -> list[Path]:
    return sorted(
        p for p in list_images(directory)
        if p.name.startswith("merged_")
    )


def get_final_merged_images(directory: Path) -> list[Path]:
    merged_images = list_merged_images(directory)
    if len(merged_images) <= 1:
        return merged_images

    final_images: list[Path] = []
    for path in merged_images:
        stem = path.stem
        is_intermediate = any(
            other != path and stem in other.stem
            for other in merged_images
        )
        if not is_intermediate:
            final_images.append(path)

    return final_images if final_images else [max(merged_images, key=lambda p: len(p.stem))]


def compare_with_baseline(
    cropped_path: Path,
    baseline_path: Path,
    output_path: Path,
) -> bool:
    if not cropped_path.exists():
        print(f"Brak pliku cropped: {cropped_path}")
        return False
    if not baseline_path.exists():
        print(f"Brak pliku baseline: {baseline_path}")
        return False

    cropped = load_rgba(cropped_path)
    baseline = load_rgba(baseline_path)

    if cropped.shape != baseline.shape:
        print(
            f"Różne wymiary ({cropped_path.name} vs {baseline_path.name}): "
            f"cropped {cropped.shape[1]}x{cropped.shape[0]} "
            f"vs baseline {baseline.shape[1]}x{baseline.shape[0]}"
        )

    h = min(cropped.shape[0], baseline.shape[0])
    w = min(cropped.shape[1], baseline.shape[1])

    diff_img = cropped.copy()
    diff_mask = np.zeros(cropped.shape[:2], dtype=bool)
    diff_mask[:h, :w] = np.any(cropped[:h, :w] != baseline[:h, :w], axis=2)

    if cropped.shape[0] > h:
        diff_mask[h:, :] = True
    if cropped.shape[1] > w:
        diff_mask[:, w:] = True

    diff_img[diff_mask] = (255, 0, 0, 255)

    diff_count = int(np.sum(diff_mask))
    total = cropped.shape[0] * cropped.shape[1]
    pct = 100.0 * diff_count / total if total else 0.0
    print(
        f"Porównanie {cropped_path.name} vs {baseline_path.name}: "
        f"{diff_count}/{total} pikseli różnych ({pct:.4f}%)"
    )

    save_rgba(output_path, diff_img)
    print(f"Zapisano mapę różnic: {output_path}")
    return diff_count == 0


def compare_cropped_with_baseline(
    work_dir: Path,
    baseline_dir: Path | None = None,
    cropped_dir: Path | None = None,
    diff_dir: Path | None = None,
) -> None:
    baseline_dir = baseline_dir or Path("baseline")
    cropped_dir = cropped_dir or work_dir / "cropped"
    diff_dir = diff_dir or Path("diff")

    baseline_images = list_images(baseline_dir)
    if not baseline_images:
        print(f"Brak obrazów baseline w {baseline_dir}")
        return

    merged_images = get_final_merged_images(work_dir)
    if not merged_images:
        print(f"Brak plików merged_* do porównania z baseline w {work_dir}")
        return

    for image_path in merged_images:
        cropped_path = cropped_dir / image_path.name
        for baseline_path in baseline_images:
            output_name = f"{cropped_path.stem}__vs__{baseline_path.stem}.png"
            compare_with_baseline(
                cropped_path,
                baseline_path,
                diff_dir / output_name,
            )


def prepare_diff_dir(diff_dir: Path) -> None:
    if diff_dir.exists():
        shutil.rmtree(diff_dir)
    diff_dir.mkdir(parents=True)


def png_to_searchable_pdf(input_path: Path, output_path: Path, lang: str = "pol") -> bool:
    if not input_path.exists():
        print(f"Nie znaleziono pliku: {input_path}")
        return False

    try:
        print(f"Trwa OCR i tworzenie PDF: {input_path.name}...")
        img = Image.open(input_path)
        pdf_bytes = pytesseract.image_to_pdf_or_hocr(img, extension="pdf", lang=lang)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(pdf_bytes)
        print(f"Utworzono przeszukiwalny PDF: {output_path}")
        return True
    except pytesseract.TesseractNotFoundError:
        print("Błąd: nie znaleziono programu Tesseract w systemie.")
        return False
    except Exception as e:
        print(f"Błąd podczas tworzenia PDF: {e}")
        return False


def convert_cropped_merges_to_pdf(work_dir: Path, cropped_dir: Path | None = None, lang: str = "pol") -> None:
    cropped_dir = cropped_dir or work_dir / "cropped"
    merged_images = get_final_merged_images(work_dir)

    if not merged_images:
        print(f"Brak plików merged_* do konwersji na PDF w {work_dir}")
        return

    for image_path in merged_images:
        png_path = cropped_dir / image_path.name
        pdf_path = cropped_dir / f"{image_path.stem}.pdf"
        png_to_searchable_pdf(png_path, pdf_path, lang=lang)


def crop_final_merges(work_dir: Path, cropped_dir: Path | None = None) -> None:
    cropped_dir = cropped_dir or work_dir / "cropped"
    merged_images = get_final_merged_images(work_dir)

    if not merged_images:
        print(f"Brak plików merged_* w {work_dir}")
        return

    print(f"Przycinam {len(merged_images)} końcowy(ch) merge:")
    for image_path in merged_images:
        print(f"  - {image_path.name}")
        crop_to_content(image_path, cropped_dir / image_path.name)


def run_merge_pipeline(in_dir: Path, out_dir: Path) -> None:
    prepare_work_dir(in_dir, out_dir)

    inputs = list(get_img_map(list_images(out_dir)).items())

    while look_for_first_match(inputs, out_dir):
        print(f"Pozostało obrazów: {len(inputs)}\n")

    print("Brak kolejnych dopasowań do zmergowania.")



def main() -> None:
    in_dir = Path("in")
    in_dir.mkdir(parents=True, exist_ok=True)

    out_dir = Path("out")
    out_dir.mkdir(parents=True, exist_ok=True)

    diff_dir = Path("diff")
    prepare_diff_dir(diff_dir)

    # run_merge_pipeline(in_dir, out_dir)

    # crop_final_merges(out_dir)

    compare_cropped_with_baseline(out_dir, diff_dir=diff_dir)

    convert_cropped_merges_to_pdf(out_dir)



if __name__ == "__main__":
    main()
