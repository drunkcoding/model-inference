import sys
import os
import shutil


from hfutils.arg_parser import DeployArguments
from hfutils.logger import Logger

logger = Logger(__file__, "info", 0, 0)

args = DeployArguments()
ins_args = args.ins_args
# model path in xxx/model/task/checkpoint-xxx/
model_path = args.model_args.model_name_or_path.strip(os.path.sep)
segment, ckpt_name = os.path.split(model_path)
segment, task_name = os.path.split(segment)
segment, model_name = os.path.split(segment)

logger.info(
    "model_path %s => model_name %s, task_name %s, update config %, update model %s",
    model_path,
    model_name,
    task_name,
    ins_args.update_cfg,
    ins_args.update_model
)

repo_path = "_".join([model_name, task_name])
repo_path = os.path.join("repository", repo_path)

if ins_args.update_cfg:
    with open(os.path.join("service", "config.pbtxt"), "r") as fp:
        cfg_text = fp.read()
        cfg_text = cfg_text.replace("MODEL", model_name)
        cfg_text = cfg_text.replace("TASK", task_name)
        cfg_text = cfg_text.replace("TOKEN_INPUT_DIM", f"-1,{args.data_args.max_length}")
        cfg_text = cfg_text.replace("SCORE_OUTPUT_DIM", f"-1,{args.data_args.num_labels}")
        cfg_text = cfg_text.replace("NUM_REPLICA", str(ins_args.num_replica))
        cfg_text = cfg_text.replace("DEVICE_MAP", ",".join(ins_args.device_map))
        cfg_text = cfg_text.replace("NUM_ENSEMBLE", "4")

        logger.info("cfg_text %s", cfg_text)

    with open(os.path.join(repo_path, "config.pbtxt"), "w") as target:
        target.write(cfg_text)

if ins_args.update_model:
    try:
        version = 0
        os.mkdir(repo_path)
        os.mkdir(os.path.join(repo_path, str(version)))
        logger.info("create repo_path %s", repo_path)
    except FileExistsError as error:
        version_list = []
        for file in os.listdir(repo_path):
            if os.path.isdir(os.path.join(repo_path, file)):
                version_list.append(file)
                version = max(version, int(file))
        version_list.sort()
        version += 1
        os.mkdir(os.path.join(repo_path, str(version)))
        logger.info("create model_repo %s, version %s", repo_path, version)

        if len(version_list) > 3:
            for v in version_list[:-2]:
                shutil.rmtree(os.path.join(repo_path, v))

    repo_path = os.path.join(repo_path, str(version))
    os.symlink(
        os.path.abspath(os.path.join("service", "model.py")),
        os.path.join(os.path.abspath(repo_path), "model.py"),
    )
    os.symlink(
        os.path.abspath(model_path),
        os.path.join(os.path.abspath(repo_path), "model"),
    )
