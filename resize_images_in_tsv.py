#!/usr/bin/env python3
"""
Resize images in TSV file to maximum width while preserving aspect ratio.

Usage:
    python resize_images_in_tsv.py input.tsv output.tsv --max-width 768
    python resize_images_in_tsv.py input.tsv output.tsv --max-width 768 --max-height 1024
"""

import sys
import pandas as pd
import argparse
import base64
from PIL import Image as PILImage
from io import BytesIO
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


def resize_image_preserve_aspect(img, max_width=768, max_height=None):
    """
    Resize image to maximum width while preserving aspect ratio.

    Args:
        img: PIL Image object
        max_width: Maximum width in pixels
        max_height: Optional maximum height in pixels

    Returns:
        PIL Image object (resized if needed)
    """
    width, height = img.size

    # Check if resizing is needed
    needs_resize = False
    new_width, new_height = width, height

    # Calculate scaling based on width constraint
    if width > max_width:
        scale = max_width / width
        new_width = max_width
        new_height = int(height * scale)
        needs_resize = True

    # Also check height constraint if specified
    if max_height and new_height > max_height:
        scale = max_height / new_height
        new_height = max_height
        new_width = int(new_width * scale)
        needs_resize = True

    # Resize if needed
    if needs_resize:
        # Use high-quality downsampling
        img = img.resize((new_width, new_height), PILImage.Resampling.LANCZOS)

    return img, needs_resize


def encode_image_to_base64(img, format='PNG', quality=95):
    """
    Encode PIL Image to base64 string.

    Args:
        img: PIL Image object
        format: Image format ('PNG' or 'JPEG')
        quality: JPEG quality (1-100, only for JPEG)

    Returns:
        Base64 encoded string
    """
    buffer = BytesIO()

    # Convert RGBA to RGB if saving as JPEG
    if format == 'JPEG' and img.mode in ('RGBA', 'LA', 'P'):
        # Create a white background
        background = PILImage.new('RGB', img.size, (255, 255, 255))
        if img.mode == 'P':
            img = img.convert('RGBA')
        background.paste(img, mask=img.split()[-1] if img.mode == 'RGBA' else None)
        img = background

    # Save to buffer
    save_kwargs = {'format': format}
    if format == 'JPEG':
        save_kwargs['quality'] = quality
        save_kwargs['optimize'] = True

    img.save(buffer, **save_kwargs)
    img_bytes = buffer.getvalue()

    return base64.b64encode(img_bytes).decode('utf-8')


def plot_size_histogram(original_sizes, resized_sizes, output_path):
    """Plot histogram comparing original and resized image dimensions."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Extract widths and heights
    orig_widths = [s[0] for s in original_sizes]
    orig_heights = [s[1] for s in original_sizes]
    resized_widths = [s[0] for s in resized_sizes]
    resized_heights = [s[1] for s in resized_sizes]

    # Width comparison
    ax = axes[0, 0]
    ax.hist([orig_widths, resized_widths], bins=30, label=['Original', 'Resized'], alpha=0.7)
    ax.axvline(768, color='red', linestyle='--', label='Max width (768px)')
    ax.set_xlabel('Width (pixels)')
    ax.set_ylabel('Frequency')
    ax.set_title('Width Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Height comparison
    ax = axes[0, 1]
    ax.hist([orig_heights, resized_heights], bins=30, label=['Original', 'Resized'], alpha=0.7)
    ax.set_xlabel('Height (pixels)')
    ax.set_ylabel('Frequency')
    ax.set_title('Height Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Aspect ratio comparison
    orig_aspects = [w/h if h > 0 else 0 for w, h in original_sizes]
    resized_aspects = [w/h if h > 0 else 0 for w, h in resized_sizes]

    ax = axes[1, 0]
    ax.hist([orig_aspects, resized_aspects], bins=30, label=['Original', 'Resized'], alpha=0.7)
    ax.set_xlabel('Aspect Ratio (width/height)')
    ax.set_ylabel('Frequency')
    ax.set_title('Aspect Ratio Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Size reduction scatter plot
    ax = axes[1, 1]
    orig_pixels = [w*h for w, h in original_sizes]
    resized_pixels = [w*h for w, h in resized_sizes]

    # Only plot images that were actually resized
    resized_indices = [i for i in range(len(orig_pixels)) if orig_pixels[i] != resized_pixels[i]]
    if resized_indices:
        orig_subset = [orig_pixels[i] for i in resized_indices]
        resized_subset = [resized_pixels[i] for i in resized_indices]

        ax.scatter(orig_subset, resized_subset, alpha=0.5, s=10)
        ax.plot([0, max(orig_pixels)], [0, max(orig_pixels)], 'r--', label='No change')
        ax.set_xlabel('Original Size (pixels²)')
        ax.set_ylabel('Resized Size (pixels²)')
        ax.set_title(f'Size Reduction ({len(resized_indices)} images resized)')
        ax.legend()
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'No images were resized', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Size Reduction')

    plt.tight_layout()

    # Save figure
    histogram_path = output_path.replace('.tsv', '_size_histogram.png')
    plt.savefig(histogram_path, dpi=100, bbox_inches='tight')
    print(f"Histogram saved to {histogram_path}")

    try:
        plt.show()
    except:
        pass


def resize_images_in_tsv(tsv_path, output_path, max_width=768, max_height=None,
                        format='PNG', quality=95, show_histogram=True):
    """
    Resize images in TSV file to maximum dimensions.

    Args:
        tsv_path: Path to input TSV file
        output_path: Path to output TSV file
        max_width: Maximum width in pixels (default: 768)
        max_height: Optional maximum height in pixels
        format: Output image format ('PNG' or 'JPEG')
        quality: JPEG quality (1-100)
        show_histogram: Whether to generate size distribution histogram
    """
    # Read the input TSV
    print(f"Reading TSV from {tsv_path}...")
    df = pd.read_csv(tsv_path, sep='\t')
    print(f"Total rows: {len(df)}")

    # Check if image column exists
    if 'image' not in df.columns:
        print("Error: 'image' column not found in TSV file")
        print(f"Available columns: {list(df.columns)}")
        sys.exit(1)

    # Process images
    print(f"\nProcessing images (max width: {max_width}px, max height: {max_height or 'unlimited'}px)...")

    original_sizes = []
    resized_sizes = []
    original_bytes = []
    resized_bytes = []
    resize_count = 0
    error_count = 0

    # Create a copy of the dataframe for modifications
    df_output = df.copy()

    # Process each row with progress bar
    for idx in tqdm(range(len(df)), desc="Resizing images"):
        row = df.iloc[idx]
        base64_str = row['image']

        if pd.notna(base64_str) and isinstance(base64_str, str):
            try:
                # Decode base64 to PIL Image
                original_bytes_data = base64.b64decode(base64_str)
                original_bytes.append(len(original_bytes_data))

                img = PILImage.open(BytesIO(original_bytes_data))
                original_size = img.size
                original_sizes.append(original_size)

                # Resize if needed
                img_resized, was_resized = resize_image_preserve_aspect(img, max_width, max_height)
                resized_sizes.append(img_resized.size)

                if was_resized:
                    resize_count += 1

                # Encode back to base64
                new_base64 = encode_image_to_base64(img_resized, format=format, quality=quality)
                resized_bytes.append(len(base64.b64decode(new_base64)))

                # Update the dataframe
                df_output.at[idx, 'image'] = new_base64

            except Exception as e:
                print(f"\nError processing image at index {row.get('index', idx)}: {e}")
                error_count += 1
                # Keep original on error
                original_sizes.append((0, 0))
                resized_sizes.append((0, 0))
                original_bytes.append(0)
                resized_bytes.append(0)
        else:
            # No image data
            original_sizes.append((0, 0))
            resized_sizes.append((0, 0))
            original_bytes.append(0)
            resized_bytes.append(0)

    # Calculate statistics
    print("\n" + "="*60)
    print("IMAGE TRANSFORMATION STATISTICS")
    print("="*60)

    # Size statistics
    valid_originals = [s for s in original_sizes if s != (0, 0)]
    valid_resized = [s for s in resized_sizes if s != (0, 0)]

    if valid_originals:
        orig_widths = [s[0] for s in valid_originals]
        orig_heights = [s[1] for s in valid_originals]
        resized_widths = [s[0] for s in valid_resized]
        resized_heights = [s[1] for s in valid_resized]

        print(f"\nTotal images processed: {len(valid_originals)}")
        print(f"Images resized: {resize_count} ({resize_count/len(valid_originals)*100:.1f}%)")
        print(f"Processing errors: {error_count}")

        print(f"\nOriginal dimensions:")
        print(f"  Width:  min={min(orig_widths):4d}px, max={max(orig_widths):4d}px, "
              f"mean={np.mean(orig_widths):6.1f}px, median={np.median(orig_widths):6.1f}px")
        print(f"  Height: min={min(orig_heights):4d}px, max={max(orig_heights):4d}px, "
              f"mean={np.mean(orig_heights):6.1f}px, median={np.median(orig_heights):6.1f}px")

        print(f"\nResized dimensions:")
        print(f"  Width:  min={min(resized_widths):4d}px, max={max(resized_widths):4d}px, "
              f"mean={np.mean(resized_widths):6.1f}px, median={np.median(resized_widths):6.1f}px")
        print(f"  Height: min={min(resized_heights):4d}px, max={max(resized_heights):4d}px, "
              f"mean={np.mean(resized_heights):6.1f}px, median={np.median(resized_heights):6.1f}px")

        # File size statistics
        valid_orig_bytes = [b for b in original_bytes if b > 0]
        valid_resized_bytes = [b for b in resized_bytes if b > 0]

        if valid_orig_bytes and valid_resized_bytes:
            orig_mb = sum(valid_orig_bytes) / (1024 * 1024)
            resized_mb = sum(valid_resized_bytes) / (1024 * 1024)

            print(f"\nFile size statistics:")
            print(f"  Original total: {orig_mb:.2f} MB")
            print(f"  Resized total:  {resized_mb:.2f} MB")
            print(f"  Size reduction: {orig_mb - resized_mb:.2f} MB ({(1 - resized_mb/orig_mb)*100:.1f}%)")
            print(f"  Average original: {np.mean(valid_orig_bytes)/1024:.1f} KB")
            print(f"  Average resized:  {np.mean(valid_resized_bytes)/1024:.1f} KB")

        # Images that exceeded limits
        exceeded_width = sum(1 for w in orig_widths if w > max_width)
        print(f"\nImages exceeding {max_width}px width: {exceeded_width} ({exceeded_width/len(orig_widths)*100:.1f}%)")

        if max_height:
            exceeded_height = sum(1 for h in orig_heights if h > max_height)
            print(f"Images exceeding {max_height}px height: {exceeded_height} ({exceeded_height/len(orig_heights)*100:.1f}%)")

    # Save the output TSV
    print(f"\nSaving transformed TSV to {output_path}...")
    df_output.to_csv(output_path, sep='\t', index=False)
    print(f"Successfully saved {len(df_output)} rows to {output_path}")

    # Generate histogram if requested
    if show_histogram and valid_originals:
        plot_size_histogram(original_sizes, resized_sizes, output_path)

    print("\n✅ Image transformation completed successfully!")
    print("   Resized images preserve aspect ratios while staying within specified limits.")
    print("   This optimization reduces memory usage and improves training efficiency.")

    return df_output


def main():
    parser = argparse.ArgumentParser(
        description='Resize images in TSV file to maximum width while preserving aspect ratio',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Resize to max 768px width (default)
  python resize_images_in_tsv.py input.tsv output.tsv

  # Custom max width
  python resize_images_in_tsv.py input.tsv output.tsv --max-width 512

  # Limit both width and height
  python resize_images_in_tsv.py input.tsv output.tsv --max-width 768 --max-height 1024

  # Use JPEG format for smaller file sizes
  python resize_images_in_tsv.py input.tsv output.tsv --format JPEG --quality 90
        """
    )

    parser.add_argument('input_file', help='Path to input TSV file')
    parser.add_argument('output_file', help='Path to output TSV file')
    parser.add_argument('--max-width', type=int, default=768,
                       help='Maximum width in pixels (default: 768)')
    parser.add_argument('--max-height', type=int, default=None,
                       help='Maximum height in pixels (optional)')
    parser.add_argument('--format', choices=['PNG', 'JPEG'], default='PNG',
                       help='Output image format (default: PNG)')
    parser.add_argument('--quality', type=int, default=95,
                       help='JPEG quality 1-100 (default: 95, only for JPEG format)')
    parser.add_argument('--no-histogram', action='store_true',
                       help='Skip generating size distribution histogram')

    args = parser.parse_args()

    # Validate arguments
    if args.max_width <= 0:
        print("Error: max-width must be positive")
        sys.exit(1)

    if args.max_height is not None and args.max_height <= 0:
        print("Error: max-height must be positive")
        sys.exit(1)

    if args.quality < 1 or args.quality > 100:
        print("Error: quality must be between 1 and 100")
        sys.exit(1)

    try:
        resize_images_in_tsv(
            args.input_file,
            args.output_file,
            max_width=args.max_width,
            max_height=args.max_height,
            format=args.format,
            quality=args.quality,
            show_histogram=not args.no_histogram
        )
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()