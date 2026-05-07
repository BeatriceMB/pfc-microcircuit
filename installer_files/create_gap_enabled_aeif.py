#!/usr/bin/env python

import matplotlib.pyplot as plt
import nest
import numpy as np
import os

from pynestml.frontend.pynestml_frontend import generate_nest_target

neuron_model = "aeif_cond_beta"
codegen_opts = {"gap_junctions": {"enable": True,
                                  "gap_current_port": "I_stim",
                                  "membrane_potential_variable": "V_m"}}

files = [os.path.join("models", "neurons", neuron_model + ".nestml")]
input_path = ["/Users/beatricebaldi/miniforge3/envs/nest-pfc/models/neurons/aeif_cond_beta.nestml"] 

generate_nest_target(input_path=input_path,
                     target_path="/Users/beatricebaldi/miniforge3/envs/nest-pfc/models/aeif_beta_gap_component",
                     logging_level="DEBUG",
                     module_name="nestml_gap_aeif_beta_module", 
                     suffix="_nestml",
                     codegen_opts=codegen_opts)
