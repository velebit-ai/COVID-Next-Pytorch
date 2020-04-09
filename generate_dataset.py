"""
Generates the COVIDx dataset from the following sources:
 * https://github.com/ieee8023/covid-chestxray-dataset.git
 * https://www.kaggle.com/c/rsna-pneumonia-detection-challenge/data

Code inspired by:
 https://github.com/lindawangg/COVID-Net/blob/master/create_COVIDx_v2.ipynb
"""

import logging
import os
from shutil import copyfile
import argparse

import numpy as np
import pandas as pd
import pydicom as dicom
from PIL import Image

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def write_metadata(pth, data):
    with open(pth, "w") as file:
        for patient_id, filename, category in data:
            info = "{} {} {}\n".format(patient_id, filename, category)
            file.write(info)


def main(args):
    train = []
    test = []
    test_count = {'normal': 0, 'pneumonia': 0, 'COVID-19': 0}
    train_count = {'normal': 0, 'pneumonia': 0, 'COVID-19': 0}

    # Create export test and train dirs
    TEST_EXPORT = os.path.join(args.save_path, 'test')
    os.makedirs(TEST_EXPORT, exist_ok=True)
    TRAIN_EXPORT = os.path.join(args.save_path, 'train')
    os.makedirs(TRAIN_EXPORT, exist_ok=True)

    mapping = dict()
    mapping['COVID-19'] = 'COVID-19'
    mapping['SARS'] = 'pneumonia'
    mapping['MERS'] = 'pneumonia'
    mapping['Streptococcus'] = 'pneumonia'
    mapping['Normal'] = 'normal'
    mapping['Lung Opacity'] = 'pneumonia'
    mapping['1'] = 'pneumonia'

    covid_imgs = os.path.join(args.covid_dir, "images")
    covid_csv = os.path.join(args.covid_dir, "metadata.csv")

    csv = pd.read_csv(covid_csv, nrows=None)
    idx_pa = csv["view"] == "PA"
    csv = csv[idx_pa]
    pneumonias = ["COVID-19", "SARS", "MERS", "ARDS", "Streptococcus"]
    pathologies = ["Pneumonia", "Viral Pneumonia", "Bacterial Pneumonia",
                   "No Finding"] + pneumonias
    pathologies = sorted(pathologies)

    filename_label = {'normal': [], 'pneumonia': [], 'COVID-19': []}
    count = {'normal': 0, 'pneumonia': 0, 'COVID-19': 0}
    for index, row in csv.iterrows():
        f = row['finding']
        if f in mapping:
            count[mapping[f]] += 1
            entry = [int(row['patientid']), row['filename'], mapping[f]]
            filename_label[mapping[f]].append(entry)

    log.info('Data distribution from covid-chestxray-dataset:')
    log.info(count)

    # add covid-chestxray-dataset into COVIDx dataset
    for key in filename_label.keys():
        arr = np.array(filename_label[key])
        if arr.size == 0:
            continue

        # Randomly sample test set patients
        patient_ids = np.unique(arr[:, 0])
        test_size = int(len(patient_ids) * args.test_size)
        test_patients = np.random.choice(patient_ids, test_size, replace=False)
        log.info('Category: {}, N test patients'.format(key, test_size))

        # go through all the patients
        for patient in arr:
            src_img_pth = os.path.join(covid_imgs, patient[1])
            if patient[0] in test_patients:
                dst_img_pth = os.path.join(TEST_EXPORT, patient[1])
                copyfile(src_img_pth, dst_img_pth)
                test.append(patient)
                test_count[patient[2]] += 1
            else:
                dst_img_pth = os.path.join(TRAIN_EXPORT, patient[1])
                copyfile(src_img_pth, dst_img_pth)
                train.append(patient)
                train_count[patient[2]] += 1

    log.info('test count: {}'.format(test_count))
    log.info('train count: {}'.format(train_count))

    # add normal and rest of pneumonia cases from
    # https://www.kaggle.com/c/rsna-pneumonia-detection-challenge
    kaggle_csv_normal = os.path.join(args.kaggle_data,
                                     "stage_2_detailed_class_info.csv")
    kaggle_csv_pneu = os.path.join(args.kaggle_data,
                                   "stage_2_train_labels.csv")
    csv_normal = pd.read_csv(kaggle_csv_normal, nrows=None)
    csv_pneu = pd.read_csv(kaggle_csv_pneu, nrows=None)
    patients = {'normal': [], 'pneumonia': []}

    for index, row in csv_normal.iterrows():
        if row['class'] == 'Normal':
            patients['normal'].append(row['patientId'])

    for index, row in csv_pneu.iterrows():
        if int(row['Target']) == 1:
            patients['pneumonia'].append(row['patientId'])

    log.info("Preparing Kaggle dataset...")
    counter = 0
    for key in patients.keys():
        arr = np.array(patients[key])
        if arr.size == 0:
            continue

        # Choose random test patients
        patient_ids = np.unique(arr)
        test_size = int(len(patient_ids) * args.test_size)
        test_patients = np.random.choice(patient_ids, test_size, replace=False)
        log.info('Category: {}, N Test examples: {}'.format(key, test_size))

        for patient in arr:
            ds = dicom.dcmread(os.path.join(args.kaggle_data,
                                            "stage_2_train_images",
                                            patient + '.dcm'))
            pixel_array_numpy = ds.pixel_array
            imgname = patient + '.png'
            pil_img = Image.fromarray(pixel_array_numpy)

            if patient in test_patients:
                pil_img.save(os.path.join(TEST_EXPORT, imgname))
                test.append([patient, imgname, key])
                test_count[key] += 1
            else:
                pil_img.save(os.path.join(TRAIN_EXPORT, imgname))
                train.append([patient, imgname, key])
                train_count[key] += 1
            counter += 1

            if counter % 500 == 0 and counter > 0:
                log.info("Converted {} Kaggle dataset images".format(counter))

    log.info('test count: {}'.format(test_count))
    log.info('train count: {}'.format(train_count))

    write_metadata(os.path.join(args.save_path, 'train_metadata.txt'), train)
    write_metadata(os.path.join(args.save_path, 'test_metadata.txt'), test)


if __name__ == "__main__":
    np.random.seed(1337)
    parser = argparse.ArgumentParser()

    parser.add_argument('--covid-dir',
                        help="Path to the cloned `covid-chestxray-dataset` "
                             "repo dir",
                        type=str)
    parser.add_argument('--kaggle-data',
                        help="Path to the downloaded Kaggle dataset dir",
                        type=str)
    parser.add_argument('--save-path',
                        help="Directory where to save the new COVIDx dataset",
                        type=str)
    parser.add_argument('--test-size',
                        help="Test set size fraction. Defaults to 10%.",
                        default=0.1,
                        type=float)
    args = parser.parse_args()
    if args.test_size < 0 or args.test_size > 1:
        raise ValueError("Test fraction value must be in range [0, 1]")
    main(args)
