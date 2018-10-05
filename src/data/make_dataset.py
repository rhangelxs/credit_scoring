# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv

import pandas as pd

from src.features.build_features import *

csv_filename = "scoring_test_task.csv"
date_columns = ["rep_loan_date", "first_loan", "first_overdue_date"]

def read_csv(*args, **kwargs):
    df = pd.read_csv(args[0])
    df = pd.read_csv(args[0], parse_dates=list(df.select_dtypes(include='object').columns))
    df = fix_nans(df)
    return df


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    # Make interim
    df = pd.read_csv("data/raw/%s" % csv_filename, parse_dates=date_columns)

    # Build features
    df = fix_trader_key(df)
    df = fix_nans(df)
    # df = fix_bad_flag(df)
    df = fix_nonunique(df)
    df = fix_dpd(df)
    df = fix_past_billings_cnt(df)
    df = fix_scores(df)
    df = fix_dates(df)

    df.to_csv("data/interim/%s" % csv_filename)

    # New
    df = generate_dpd(df)
    df = generate_dates(df)

    # Write final
    df.to_csv("data/processed/%s" % csv_filename)

    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
