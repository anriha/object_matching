import argparse

parser = argparse.ArgumentParser(description='reid')

parser.add_argument("--data_path",
                    help="paths to datasets",
                    nargs="+",
                    required=True)

parser.add_argument("--weight",
                    help="path to pretrained model", 
                    required=True)

parser.add_argument("--output_file",
                    help="path to output csv",
                    required=True)

parser.add_argument("--embeddings_path",
                    help="path to save generated embeddings")

opt = parser.parse_args()
