from annotation.Convert_data import RetailProductDatasetConverter
import yaml

def prepare_data(converter):
    converter.convert_annotation('train')
    converter.convert_annotation('val')
    converter.convert_annotation('test')

    detect_yaml = dict(
        train=converter.train_path,
        val=converter.val_path,
        nc=converter.nc,
        names=converter.names
    )

    with open('detect.yaml', 'w') as outfile:
        yaml.dump(detect_yaml, outfile, default_flow_style=True)

def main():
    # Replace these with the actual paths
    data_root = '../datasets/retail_product_checkout_dataset'
    work_root = '../datasets/retail_product_checkout_dataset_labels'

    converter = RetailProductDatasetConverter(data_root, work_root)
    prepare_data(converter)

if __name__ == "__main__":
    main()
