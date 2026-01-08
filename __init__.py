"""
CASP Project Root Package
"""

__version__ = "1.0.0"
__author__ = "Yu Dai"
__email__ = "2020141530133@stu.scu.edu.cn"

from config.config import CASPConfig
from casp_pipeline import CASPPipeline

__all__ = ['CASPConfig', 'CASPPipeline']
