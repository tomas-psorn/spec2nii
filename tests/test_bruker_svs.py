import pytest
import numpy as np
import os.path as op
import subprocess
from fsl_mrs.utils import mrs_io
from PIL import Image,ImageDraw,ImageFont
from spec2nii.spec2nii import spec2nii

bruker_path = '/home/tomas/data/20200612_094625_lego_phantom_3_1_2'

svs_data_names_fid_601 = ['35/fid', '36/fid']

spec2nii
