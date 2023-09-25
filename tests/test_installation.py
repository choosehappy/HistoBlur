import os
import platform
import re
from subprocess import getstatusoutput, getoutput

PRG = 'HistoBlur'

def test_general_help_works() -> None:
    """ -h option prints help page """
    rv, out = getstatusoutput('{} -h'.format(PRG))
    assert rv == 0
    assert out.lower().startswith('usage:')

def test_train_help_works() -> None:
    """ -h option prints help page """
    rv, out = getstatusoutput('{} train -h'.format(PRG))
    assert rv == 0
    assert out.lower().startswith('usage:')

def test_detect_help_works() -> None:
    """ -h option prints help page """
    rv, out = getstatusoutput('{} detect -h'.format(PRG))
    assert rv == 0
    assert out.lower().startswith('usage:')