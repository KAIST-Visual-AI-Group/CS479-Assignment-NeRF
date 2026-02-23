"""
resize_images.py

A script that resizes images in the Blender dataset to a given size.
"""

import argparse
from pathlib import Path
import shutil

import cv2


def parse_args() -> argparse.Namespace:
    """
    Parses command-line arguments.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_path", type=str, help="Path to Blender dataset")
    parser.add_argument(
        "--target_size",
        type=int,
        default=200,
        help="Target image size. Images are resized to (target_size, target_size).",
    )

    args = parser.parse_args()
    return args


def main(args: argparse.Namespace) -> None:
    """The entrypoint of the script"""
    
    # =========================================================================
    # parse arguments
    dataset_path = Path(args.dataset_path).resolve()
    target_size = args.target_size
    # =========================================================================

    # =========================================================================
    # create output directory
    dataset_name = dataset_path.stem
    output_dir = dataset_path.parent / f"{dataset_name}_resized_{target_size}"
    output_dir.mkdir(exist_ok=True, parents=True)
    print(f"[!] Created directory: {output_dir}")
    # =========================================================================


    # =========================================================================
    # copy metadata files
    metadata_train = dataset_path / "transforms_train.json"
    assert metadata_train.exists(), f"File {str(metadata_train)} does not exist"
    metadata_val = dataset_path / "transforms_val.json"
    assert metadata_val.exists(), f"File {str(metadata_val)} does not exist"
    metadata_test = dataset_path / "transforms_test.json"
    assert metadata_test.exists(), f"File {str(metadata_test)} does not exist"
    
    shutil.copy(metadata_train, output_dir)
    shutil.copy(metadata_val, output_dir)
    shutil.copy(metadata_test, output_dir)
    # =========================================================================

    # =========================================================================
    # copy images
    images_train = dataset_path / "train"
    assert images_train.exists(), f"Directory {str(images_train)} does not exist"
    images_val = dataset_path / "val"
    assert images_val.exists(), f"Directory {str(images_val)} does not exist"
    images_test = dataset_path / "test"
    assert images_test.exists(), f"Directory {str(images_test)} does not exist"
    
    images_train_output = output_dir / "train"
    images_train_output.mkdir(exist_ok=True, parents=True)
    images_val_output = output_dir / "val"
    images_val_output.mkdir(exist_ok=True, parents=True)
    images_test_output = output_dir / "test"
    images_test_output.mkdir(exist_ok=True, parents=True)
    
    shutil.copytree(images_train, images_train_output, dirs_exist_ok=True)
    shutil.copytree(images_val, images_val_output, dirs_exist_ok=True)
    shutil.copytree(images_test, images_test_output, dirs_exist_ok=True)

    print("[!] Successfully copied images")
    # =========================================================================

    # =========================================================================
    # resize images
    for image_path in images_train_output.glob("*.png"):
        image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
        image = cv2.resize(
            image,
            (target_size, target_size),
            interpolation=cv2.INTER_AREA,
        )
        cv2.imwrite(str(image_path), image)

    for image_path in images_val_output.glob("*.png"):
        image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
        image = cv2.resize(
            image,
            (target_size, target_size),
            interpolation=cv2.INTER_AREA,
        )
        cv2.imwrite(str(image_path), image)

    # TODO: Handle normal maps and depth maps (not used yet)
    for image_path in images_test_output.glob("*.png"):
        image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
        image = cv2.resize(
            image,
            (target_size, target_size),
            interpolation=cv2.INTER_AREA,
        )
        cv2.imwrite(str(image_path), image)
    
    print("[!] Successfully resized images")
    # =========================================================================


if __name__ == "__main__":
    args = parse_args()
    main(args)
