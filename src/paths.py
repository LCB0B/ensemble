""" Controls Paths """

import getpass
import os
import platform
import random
import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

from src.utils import print_main


class ProjectPaths:
    """
    Class containing file paths for the project directory and its subdirectories.
    """

    def __init__(self, k_home: str = "/home/louibo/ensemble") -> None:
        """
        Initialize the ProjectPaths class with base directory.

        Args:
            k_home (str): The base directory of the project.
        """
        if platform.system() == "Linux":
            self.FPATH_PROJECT = Path(f"{k_home}")
            self.FPATH_D_DRIVE =  Path(f"{k_home}")
        else:
            self.FPATH_PROJECT = Path(rf"K:/{k_home}")
            self.FPATH_D_DRIVE = Path(rf"D:/work/{getpass.getuser()}")

        self.DATA = self.FPATH_D_DRIVE / "data" / 'data_reverted'
        self.NETWORK_DATA = self.FPATH_PROJECT / "data" / 'data_reverted'
        # TODO: Decide whether to log to local or network
        # self.CHECKPOINTS: Path = self.FPATH_D_DRIVE / "checkpoints"
        self.CHECKPOINTS = self.FPATH_PROJECT / "checkpoints"
        self.CHECKPOINTS_TRANSFORMER = self.CHECKPOINTS / "transformer"
        self.CHECKPOINTS_CATBOOST = self.CHECKPOINTS / "catboost"
        self.CHECKPOINTS_TABULAR = self.CHECKPOINTS / "tabular"
        # self.TB_LOGS: Path = self.FPATH_D_DRIVE / "tb_logs"
        self.CONFIGS = self.FPATH_PROJECT / "configs"
        self.TABLES = self.FPATH_PROJECT / "tables"
        self.FIGURES = self.FPATH_PROJECT / "figures"
        self.TEMP_FILES = self.DATA / "temp_files"
        # self.TEMP_FILES.mkdir(parents=True, exist_ok=True)
        self.DUMP_DIR = self.DATA / "dumps"
        self.NETWORK_DUMP_DIR = self.NETWORK_DATA / "dumps"
        self.LOGS = self.FPATH_PROJECT / "logs"
        self.TB_LOGS = self.LOGS / "transformer_logs"
        self.BASELINE_LOGS = self.LOGS / "tabular_logs"
        self.OPTUNA = self.LOGS / "optuna"
        self.DEFAULT_MODEL = self.CHECKPOINTS_TRANSFORMER / 'destiny' / 'model'  / 'best.ckpt'
        self.DEFAULT_HPARAMS = self.CONFIGS /'destiny' / 'hparams_destiny_pretrain.yaml'

    def swap_drives(self, fpath: Path):
        """Swaps the drive of fpath by checking parts and replacing with opposite drive"""
        if fpath.parts[: len(self.FPATH_PROJECT.parts)] == self.FPATH_PROJECT.parts:
            return self.FPATH_D_DRIVE / fpath.relative_to(self.FPATH_PROJECT)
        elif fpath.parts[: len(self.FPATH_D_DRIVE.parts)] == self.FPATH_D_DRIVE.parts:
            return self.FPATH_PROJECT / fpath.relative_to(self.FPATH_D_DRIVE)
        else:
            raise ValueError("Only supports K and D drive")

    def copy_to_opposite_drive(self, fpath: Path):
        """Copies file to the opposite drive (D->K or K->D)"""
        self.swap_drives(fpath).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(fpath, self.swap_drives(fpath))

    def alternative_copy_to_opposite_drive(self, fpath: Path) -> None:
        """
        Copies file or directory to the opposite drive (D->K or K->D).

        Args:
            fpath (Path): The path of the file or directory to copy.
        """
        dest_path = self.swap_drives(fpath)
        if fpath.is_dir():
            shutil.copytree(fpath, dest_path, dirs_exist_ok=True)
        else:
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(fpath, dest_path)


FPATH = ProjectPaths()


def check_and_copy_file(file_path: Path):
    """
    Checks if a file exists within the specified drive and copy from opposite drive if it exists there.

    Args:
        file_path (Path): The name of the file to check.
    """
    swapped_fpath = FPATH.swap_drives(file_path)
    if not file_path.exists() and not swapped_fpath.exists():
        raise FileNotFoundError(f"{file_path} does not exist on either drive.")
    elif not file_path.exists() and swapped_fpath.exists():
        print_main(f"{file_path} does not exist. Initiating copy from opposite drive.")
        FPATH.copy_to_opposite_drive(swapped_fpath)
    elif swapped_fpath.exists():
        print_main(f"{file_path} exists.")


def check_and_copy_file_or_dir(file_path: Path, verbosity=3) -> None:
    """
    Checks if a file or directory exists on the specified drive first, then the opposite drive.
    If it exists on the opposite drive, copies it to the specified drive and uses it.
    If the file or directory does not exist on either drive, returns None.

    Args:
        file_path (Path): The path of the file or directory to check and copy if needed.
        verbosity (int): 0 no prints, 1 print only existence, 2 print non-existence, 3 print all
    """
    if file_path.exists():
        if verbosity in [1, 3]:
            print(f"{file_path} exists on this drive.")
        return True

    swapped_fpath = FPATH.swap_drives(file_path)
    if swapped_fpath.exists():
        if verbosity in [2, 3]:
            print(
                f"{file_path} does not exist on this drive. Initiating copy from {swapped_fpath}"
            )
        FPATH.alternative_copy_to_opposite_drive(swapped_fpath)
        return True
    else:
        if verbosity in [2, 3]:
            print(f"{file_path} does not exist on either drive.")
        return False


def copy_file_or_dir(file_path: Path) -> None:
    """
    Copy file from opposite drive

    Args:
        file_path (Path): The path of the file or directory to check and copy if needed.
    """

    swapped_fpath = FPATH.swap_drives(file_path)
    if swapped_fpath.exists():
        print(f"Initiating copy from {swapped_fpath}")
        FPATH.alternative_copy_to_opposite_drive(swapped_fpath)
        return True
    else:
        print(f"{file_path} does not exist on opposite drive.")
        return False


def get_wandb_runid(file_dir: Path):
    """Gets a random name for a run id"""
    # fmt: off
    adjectives =['abundant', 'accessible', 'accommodating', 'accurate', 'adaptable', 'adept', 'admirable', 'adorable', 'adventurous', 'affable', 'affectionate', 'aggressive', 'agile', 'agreeable', 'agreeing', 'alert', 'alive', 'alluring', 'amazed', 'amazing', 'amiable', 'amused', 'amusing', 'angry', 'annoyed', 'annoying', 'anxious', 'apprehensive', 'arrogant', 'artful', 'artistic', 'ashamed', 'assertive', 'astonishing', 'astute', 'attentive', 'attractive', 'audacious', 'authentic', 'average', 'avid', 'awful', 'bad', 'balanced', 'beautiful', 'benevolent', 'better', 'bewildered', 'black', 'blissful', 'bloody', 'blue', 'blushing', 'bold', 'bored', 'brainy', 'brave', 'breakable', 'bright', 'brilliant', 'bubbly', 'buoyant', 'busy', 'calm', 'capable', 'carefree', 'careful', 'caring', 'cautious', 'centered', 'charismatic', 'charming', 'cheerful', 'chilly', 'classy', 'clean', 'clear', 'clever', 'cloudy', 'clumsy', 'colorful', 'colossal', 'combative', 'comfortable', 'committed', 'compassionate', 'competent', 'concerned', 'condemned', 'confident', 'confused', 'considerate', 'constructive', 'contemplative', 'content', 'convivial', 'cooperative', 'cordial', 'courageous', 'courteous', 'crafty', 'crazy', 'creative', 'credible', 'creepy', 'crowded', 'cruel', 'curious', 'curvy', 'cute', 'dangerous', 'daring', 'dark', 'dashing', 'dead', 'decisive', 'dedicated', 'defeated', 'defiant', 'deliberate', 'delightful', 'dependable', 'depressed', 'desirable', 'determined', 'devoted', 'different', 'difficult', 'dignified', 'diligent', 'diplomatic', 'direct', 'discerning', 'discreet', 'disgusted', 'distinct', 'disturbed', 'dizzy', 'doubtful', 'drab', 'dull', 'dynamic', 'eager', 'earnest', 'easy', 'effective', 'efficient', 'elated', 'elegant', 'eloquent', 'embarrassed', 'empathetic', 'enchanting', 'encouraging', 'energetic', 'engaged', 'enthusiastic', 'envious', 'ethical', 'evil', 'excited', 'exemplary', 'exotic', 'expensive', 'experienced', 'extraordinary', 'exuberant', 'fair', 'faithful', 'famous', 'fancy', 'fantastic', 'fearless', 'fierce', 'fiery', 'filthy', 'fine', 'focused', 'foolish', 'forthright', 'fragile', 'frail', 'frantic', 'friendly', 'frightened', 'funny', 'gentle', 'genuine', 'gifted', 'giving', 'glamorous', 'gleaming', 'glorious', 'good', 'gorgeous', 'graceful', 'gregarious', 'grieving', 'grotesque', 'grumpy', 'handsome', 'happy', 'hardworking', 'harmonious', 'healthy', 'helpful', 'helpless', 'heroic', 'hilarious', 'homeless', 'homely', 'honest', 'hopeful', 'horrible', 'humble', 'humorous', 'hungry', 'hurt', 'ill', 'imaginative', 'impartial', 'important', 'impossible', 'incomparable', 'independent', 'inexpensive', 'ingenious', 'innocent', 'innovative', 'inquisitive', 'inspired', 'inspiring', 'intellectual', 'intelligent', 'intuitive', 'inventive', 'inviting', 'itchy', 'jealous', 'jittery', 'jolly', 'jovial', 'joyful', 'joyous', 'judicious', 'just', 'keen', 'kind', 'knowledgeable', 'laudable', 'lazy', 'leaderly', 'lenient', 'light', 'likable', 'lively', 'logical', 'lonely', 'long', 'lovely', 'loyal', 'lucid', 'lucky', 'magical', 'magnificent', 'majestic', 'mature', 'methodical', 'meticulous', 'mighty', 'mindful', 'misty', 'modern', 'modest', 'motionless', 'motivated', 'muddy', 'mushy', 'mysterious', 'nasty', 'natural', 'naughty', 'nervous', 'nice', 'nifty', 'noble', 'nurturing', 'nutty', 'obedient', 'obnoxious', 'odd', 'open', 'optimistic', 'orderly', 'organized', 'outgoing', 'outrageous', 'outstanding', 'panicky', 'passionate', 'patient', 'peaceful', 'perceptive', 'perfect', 'persistent', 'philosophical', 'plain', 'playful', 'pleasant', 'poised', 'polished', 'poor', 'positive', 'powerful', 'practical', 'precious', 'precise', 'prickly', 'proactive', 'productive', 'proficient', 'proud', 'prudent', 'punctual', 'putrid', 'puzzled', 'quaint', 'quick', 'radiant', 'rational', 'real', 'reliable', 'relieved', 'remarkable', 'repulsive', 'resilient', 'resolute', 'resourceful', 'respectful', 'responsible', 'responsive', 'reverent', 'rich', 'robust', 'savvy', 'scary', 'scholarly', 'selfish', 'selfless', 'sensible', 'serene', 'sharp', 'shiny', 'shrewd', 'shy', 'silly', 'sincere', 'skillful', 'sleepy', 'smart', 'smiling', 'smoggy', 'sociable', 'solid', 'sore', 'sparkling', 'spirited', 'splendid', 'spontaneous', 'spotless', 'steadfast', 'steady', 'stormy', 'strange', 'strategic', 'strong', 'studious', 'stupid', 'successful', 'super', 'supportive', 'sustainable', 'sympathetic', 'tactful', 'talented', 'tame', 'tasty', 'tenacious', 'tender', 'tense', 'terrible', 'thankful', 'thoughtful', 'thoughtless', 'thriving', 'tired', 'tireless', 'tolerant', 'tough', 'troubled', 'trustworthy', 'ugliest', 'ugly', 'unassuming', 'unbiased', 'understanding', 'uninterested', 'unique', 'unsightly', 'unusual', 'upbeat', 'upset', 'uptight', 'vast', 'versatile', 'vibrant', 'victorious', 'vigilant', 'vivacious', 'vivid', 'wandering', 'warm', 'weary', 'wicked', 'wild', 'wise', 'witty', 'worried', 'worrisome', 'wrong', 'youthful', 'zany', 'zealous', 'zestful']
    nouns = ["tiger", "falcon", "whale", "dragon", "phoenix", "eagle", "panther", "lion", "wolf", "hawk", "unicorn", "griffin", "centaur", "hydra", "lynx", "serpent", "kraken", "bear", "cheetah", "orca", "rhino", "cougar", "leopard", "cobra", "jaguar", "raven", "sparrow", "sphinx", "chameleon", "pelican", "giraffe", "fox", "ocelot", "mustang", "buffalo", "stallion", "hippo", "bison", "shark", "gazelle", "otter", "penguin"]
    # fmt: on

    if len(files := list(file_dir.glob("*"))) > 0:
        run_number = int(max(files, key=lambda f: f.stat().st_mtime).stem.split("_")[0])
        run_number = str(run_number + 1).zfill(3)
    else:
        run_number = "001"
    run_id = f"{run_number}_{random.choice(adjectives)}_{random.choice(nouns)}"

    print_main("Staring run with run_id:", run_id)
    return run_id


def get_newest_path(path: Path) -> Optional[Path]:
    """
    Find the name of the newest file or folder in the specified directory.

    Args:
        path (Path): The path to the directory.

    Returns:
        Optional[str]: The name of the newest file or folder, or None if the directory is empty.
    """
    if path.is_dir():
        newest = max(path.iterdir(), key=lambda p: p.stat().st_mtime, default=None)
        if newest:
            return newest.name
    return None


@contextmanager
def local_copy(file_path: str):
    """
    Context manager to copy a file to a local temp directory and clean it up after use.

    Args
        file_path (str): (str) Path to the original file.

    Yields:
        str: Local temp file path.
    """
    temp_dir = tempfile.mkdtemp()
    local_file = os.path.join(temp_dir, os.path.basename(file_path))
    copy_file_or_dir(file_path, local_file)
    try:
        yield local_file
    finally:
        shutil.rmtree(temp_dir)


import re


def get_wandb_runid_filter_away_nonconforming(file_dir: Path) -> str:
    """
    Generates a run ID using an incrementing 3-digit number based on existing files
    in the directory that start with a 3-digit prefix followed by an underscore.

    Args:
        file_dir (Path): Directory to scan for existing runs.
    """
    pattern = re.compile(r"^(\d{3})_")

    # fmt: off
    adjectives =['abundant', 'accessible', 'accommodating', 'accurate', 'adaptable', 'adept', 'admirable', 'adorable', 'adventurous', 'affable', 'affectionate', 'aggressive', 'agile', 'agreeable', 'agreeing', 'alert', 'alive', 'alluring', 'amazed', 'amazing', 'amiable', 'amused', 'amusing', 'angry', 'annoyed', 'annoying', 'anxious', 'apprehensive', 'arrogant', 'artful', 'artistic', 'ashamed', 'assertive', 'astonishing', 'astute', 'attentive', 'attractive', 'audacious', 'authentic', 'average', 'avid', 'awful', 'bad', 'balanced', 'beautiful', 'benevolent', 'better', 'bewildered', 'black', 'blissful', 'bloody', 'blue', 'blushing', 'bold', 'bored', 'brainy', 'brave', 'breakable', 'bright', 'brilliant', 'bubbly', 'buoyant', 'busy', 'calm', 'capable', 'carefree', 'careful', 'caring', 'cautious', 'centered', 'charismatic', 'charming', 'cheerful', 'chilly', 'classy', 'clean', 'clear', 'clever', 'cloudy', 'clumsy', 'colorful', 'colossal', 'combative', 'comfortable', 'committed', 'compassionate', 'competent', 'concerned', 'condemned', 'confident', 'confused', 'considerate', 'constructive', 'contemplative', 'content', 'convivial', 'cooperative', 'cordial', 'courageous', 'courteous', 'crafty', 'crazy', 'creative', 'credible', 'creepy', 'crowded', 'cruel', 'curious', 'curvy', 'cute', 'dangerous', 'daring', 'dark', 'dashing', 'dead', 'decisive', 'dedicated', 'defeated', 'defiant', 'deliberate', 'delightful', 'dependable', 'depressed', 'desirable', 'determined', 'devoted', 'different', 'difficult', 'dignified', 'diligent', 'diplomatic', 'direct', 'discerning', 'discreet', 'disgusted', 'distinct', 'disturbed', 'dizzy', 'doubtful', 'drab', 'dull', 'dynamic', 'eager', 'earnest', 'easy', 'effective', 'efficient', 'elated', 'elegant', 'eloquent', 'embarrassed', 'empathetic', 'enchanting', 'encouraging', 'energetic', 'engaged', 'enthusiastic', 'envious', 'ethical', 'evil', 'excited', 'exemplary', 'exotic', 'expensive', 'experienced', 'extraordinary', 'exuberant', 'fair', 'faithful', 'famous', 'fancy', 'fantastic', 'fearless', 'fierce', 'fiery', 'filthy', 'fine', 'focused', 'foolish', 'forthright', 'fragile', 'frail', 'frantic', 'friendly', 'frightened', 'funny', 'gentle', 'genuine', 'gifted', 'giving', 'glamorous', 'gleaming', 'glorious', 'good', 'gorgeous', 'graceful', 'gregarious', 'grieving', 'grotesque', 'grumpy', 'handsome', 'happy', 'hardworking', 'harmonious', 'healthy', 'helpful', 'helpless', 'heroic', 'hilarious', 'homeless', 'homely', 'honest', 'hopeful', 'horrible', 'humble', 'humorous', 'hungry', 'hurt', 'ill', 'imaginative', 'impartial', 'important', 'impossible', 'incomparable', 'independent', 'inexpensive', 'ingenious', 'innocent', 'innovative', 'inquisitive', 'inspired', 'inspiring', 'intellectual', 'intelligent', 'intuitive', 'inventive', 'inviting', 'itchy', 'jealous', 'jittery', 'jolly', 'jovial', 'joyful', 'joyous', 'judicious', 'just', 'keen', 'kind', 'knowledgeable', 'laudable', 'lazy', 'leaderly', 'lenient', 'light', 'likable', 'lively', 'logical', 'lonely', 'long', 'lovely', 'loyal', 'lucid', 'lucky', 'magical', 'magnificent', 'majestic', 'mature', 'methodical', 'meticulous', 'mighty', 'mindful', 'misty', 'modern', 'modest', 'motionless', 'motivated', 'muddy', 'mushy', 'mysterious', 'nasty', 'natural', 'naughty', 'nervous', 'nice', 'nifty', 'noble', 'nurturing', 'nutty', 'obedient', 'obnoxious', 'odd', 'open', 'optimistic', 'orderly', 'organized', 'outgoing', 'outrageous', 'outstanding', 'panicky', 'passionate', 'patient', 'peaceful', 'perceptive', 'perfect', 'persistent', 'philosophical', 'plain', 'playful', 'pleasant', 'poised', 'polished', 'poor', 'positive', 'powerful', 'practical', 'precious', 'precise', 'prickly', 'proactive', 'productive', 'proficient', 'proud', 'prudent', 'punctual', 'putrid', 'puzzled', 'quaint', 'quick', 'radiant', 'rational', 'real', 'reliable', 'relieved', 'remarkable', 'repulsive', 'resilient', 'resolute', 'resourceful', 'respectful', 'responsible', 'responsive', 'reverent', 'rich', 'robust', 'savvy', 'scary', 'scholarly', 'selfish', 'selfless', 'sensible', 'serene', 'sharp', 'shiny', 'shrewd', 'shy', 'silly', 'sincere', 'skillful', 'sleepy', 'smart', 'smiling', 'smoggy', 'sociable', 'solid', 'sore', 'sparkling', 'spirited', 'splendid', 'spontaneous', 'spotless', 'steadfast', 'steady', 'stormy', 'strange', 'strategic', 'strong', 'studious', 'stupid', 'successful', 'super', 'supportive', 'sustainable', 'sympathetic', 'tactful', 'talented', 'tame', 'tasty', 'tenacious', 'tender', 'tense', 'terrible', 'thankful', 'thoughtful', 'thoughtless', 'thriving', 'tired', 'tireless', 'tolerant', 'tough', 'troubled', 'trustworthy', 'ugliest', 'ugly', 'unassuming', 'unbiased', 'understanding', 'uninterested', 'unique', 'unsightly', 'unusual', 'upbeat', 'upset', 'uptight', 'vast', 'versatile', 'vibrant', 'victorious', 'vigilant', 'vivacious', 'vivid', 'wandering', 'warm', 'weary', 'wicked', 'wild', 'wise', 'witty', 'worried', 'worrisome', 'wrong', 'youthful', 'zany', 'zealous', 'zestful']
    nouns = ["tiger", "falcon", "whale", "dragon", "phoenix", "eagle", "panther", "lion", "wolf", "hawk", "unicorn", "griffin", "centaur", "hydra", "lynx", "serpent", "kraken", "bear", "cheetah", "orca", "rhino", "cougar", "leopard", "cobra", "jaguar", "raven", "sparrow", "sphinx", "chameleon", "pelican", "giraffe", "fox", "ocelot", "mustang", "buffalo", "stallion", "hippo", "bison", "shark", "gazelle", "otter", "penguin"]
    # fmt: on

    run_numbers = []
    for f in file_dir.glob("*"):
        match = pattern.match(f.stem)
        if match:
            run_numbers.append(int(match.group(1)))

    if run_numbers:
        run_number = str(max(run_numbers) + 1).zfill(3)
    else:
        run_number = "001"

    run_id = f"{run_number}_{random.choice(adjectives)}_{random.choice(nouns)}"

    print_main("Starting run with run_id:", run_id)
    return run_id
