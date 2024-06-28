import os

# Path ke direktori yang berisi file label
annotations_path = "D:/MY_FILES/Datasets/act_gm1_revised 1/train/labels"
target_class = "1"  # Kelas yang ingin dihapus


def remove_class_from_annotations(annotations_path, target_class):
    for filename in os.listdir(annotations_path):
        if filename.endswith(".txt"):
            filepath = os.path.join(annotations_path, filename)
            with open(filepath, "r") as file:
                lines = file.readlines()

            updated_lines = [line for line in lines if not line.startswith(target_class + " ")]

            with open(filepath, "w") as file:
                file.writelines(updated_lines)
            print(f"Processed {filename}")


# Panggil fungsi untuk memperbarui semua anotasi dalam direktori
remove_class_from_annotations(annotations_path, target_class)

print(f"All annotations in {annotations_path} with class {target_class} have been removed.")
