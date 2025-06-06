import json
import logging
import click
from pathlib import Path
from shutil import make_archive
import os
import datetime


from src.utils.custom_logging import configure_logger
from src.ate.experiment import ate_experiments
from src.cate.experiment import cate_experiments
from src.dml_ci.experiment import ci_experiments
from src.me_ci.experiment import me_ci_experiments
from src.sate_ci.experiment import sate_ci_experiments
from src.me.experiment import me_experiments
from src.sate.experiment import sate_experiments


DATA_DIR = Path.cwd().joinpath('data')
DUMP_DIR = Path.cwd().joinpath('dumps')
SRC_DIR = Path.cwd().joinpath('src')
SLACK_URL = None
NUM_GPU = None

SCRIPT_NAME = Path(__file__).stem
LOG_DIR = Path.cwd().joinpath(f'logs/{SCRIPT_NAME}-{str(datetime.datetime.now().strftime("%m-%d-%H-%M-%S"))}')

logger = logging.getLogger()


@click.group()
@click.argument('config_path', type=click.Path(exists=True))
@click.option('--debug/--release', default=False)
@click.pass_context
def main(ctx, config_path, debug):
    if(debug):
        # Change logging level to debug
        logger.setLevel(logging.DEBUG)
        logger.handlers[-1].setLevel(logging.DEBUG)
        logger.debug("debug")

    foldername = str(datetime.datetime.now().strftime("%m-%d-%H-%M-%S"))
    dump_dir = DUMP_DIR.joinpath(foldername)
    os.mkdir(dump_dir)
    with open(config_path) as f:
        config = json.load(f)
    ctx.obj["data_dir"] = dump_dir
    ctx.obj["config"] = config
    json.dump(config, open(dump_dir.joinpath("configs.json"), "w"), indent=4)
    make_archive(dump_dir.joinpath("src"), "zip", root_dir=SRC_DIR)


@main.command()
@click.pass_context
@click.option("--num_thread", "-t", default=1, type=int)
def ate(ctx, num_thread):
    config = ctx.obj["config"]
    data_dir = ctx.obj["data_dir"]
    ate_experiments(config, data_dir, num_thread, NUM_GPU)

@main.command()
@click.pass_context
@click.option("--num_thread", "-t", default=1, type=int)
def cate(ctx, num_thread):
    config = ctx.obj["config"]
    data_dir = ctx.obj["data_dir"]
    cate_experiments(config, data_dir, num_thread, NUM_GPU)

@main.command()
@click.pass_context
@click.option("--num_thread", "-t", default=1, type=int)
def ci(ctx, num_thread):
    config = ctx.obj["config"]
    data_dir = ctx.obj["data_dir"]
    ci_experiments(config, data_dir, num_thread, NUM_GPU)

@main.command()
@click.pass_context
@click.option("--num_thread", "-t", default=1, type=int)
def me(ctx, num_thread):
    config = ctx.obj["config"]
    data_dir = ctx.obj["data_dir"]
    me_experiments(config, data_dir, num_thread, NUM_GPU)


@main.command()
@click.pass_context
@click.option("--num_thread", "-t", default=1, type=int)
def sate(ctx, num_thread):
    config = ctx.obj["config"]
    data_dir = ctx.obj["data_dir"]
    sate_experiments(config, data_dir, num_thread, NUM_GPU)
    
    
@main.command()
@click.pass_context
@click.option("--num_thread", "-t", default=1, type=int)
def meci(ctx, num_thread):
    config = ctx.obj["config"]
    data_dir = ctx.obj["data_dir"]
    me_ci_experiments(config, data_dir, num_thread, NUM_GPU)


@main.command()
@click.pass_context
@click.option("--num_thread", "-t", default=1, type=int)
def sateci(ctx, num_thread):
    config = ctx.obj["config"]
    data_dir = ctx.obj["data_dir"]
    sate_ci_experiments(config, data_dir, num_thread, NUM_GPU)


if __name__ == '__main__':
    configure_logger(SCRIPT_NAME, log_dir=LOG_DIR, webhook_url=SLACK_URL)
    try:
        main(obj={})
        logger.critical('===== Script completed successfully! =====')
    except Exception as e:
        logger.exception(e)