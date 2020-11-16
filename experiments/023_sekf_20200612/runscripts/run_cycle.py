#!/bin/env python
# -*- coding: utf-8 -*-
#
#Created on 20.03.17
#
#Created for py_bacy
#
#@author: Tobias Sebastian Finn, tobias.sebastian.finn@studium.uni-hamburg.de
#
#    Copyright (C) {2017}  {Tobias Sebastian Finn}
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <http://www.gnu.org/licenses/>.
#

# System modules
import logging
import os

# External modules

# Internal modules
from py_bacy.cycle import Cycle
from py_bacy.tsmp_restart import TSMPRestartModule
from py_bacy.tsmp_finite import TSMPFiniteModule
from py_bacy.finite_pert import FinitePertModule
from py_bacy.sekf import SEKFModule


def main():
    logging.basicConfig(level=logging.DEBUG)
    configs = '.'
    cycle = Cycle(os.path.join(configs, 'cycle.yml'))
    cycle.register_model(TSMPRestartModule, 'terrsysmp', None,
                         config_path=os.path.join(configs, 'tsmp.yml'))
    cycle.register_model(
        FinitePertModule, 'finite_pert', 'terrsysmp',
        config_path=os.path.join(configs, 'finite_pert.yml')
    )
    cycle.register_model(
        TSMPFiniteModule, 'terrsysmp_finite', 'finite_pert',
        config_path=os.path.join(configs, 'tsmp_finite.yml')
    )
    cycle.register_model(
        SEKFModule, 'sekf', 'terrsysmp_finite',
        config_path=os.path.join(configs, 'sekf.yml')
    )
    cycle.run()


if __name__ == '__main__':
    main()

