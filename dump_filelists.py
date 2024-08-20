from glob import glob
import os
import argparse

parser = argparse.ArgumentParser(description='Code to generate filelist base on data_root content')
parser.add_argument("--data_root", help="Root folder of the preprocessed LRS2 dataset", default="data_root/main")
args = parser.parse_args()


def dump_filelist(base_path):
    result = list(glob("{}/*".format(base_path)))
    result_list = []
    for i, dirpath in enumerate(result):
        dirs = os.listdir(dirpath)
        file_name = "{}.txt".format(os.path.basename(dirpath))
        for f in dirs:
            result_list.append(str(os.path.join(dirpath.replace(base_path, ''), f).replace('.mp4', '').strip('/')))
            with open(os.path.join("filelists", file_name), 'w', encoding='utf-8') as fi:
                fi.write("\n".join(result_list))


if __name__ == '__main__':
    data_root = args.data_root
    dump_filelist(data_root)
