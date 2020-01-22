# -*- coding: utf-8 -*-
"""
This file contains the setup of the neuronal network running the Husky experiment with neuronal image recognition
"""
# pragma: no cover

__author__ = 'Stefano Nardo'

from hbp_nrp_cle.brainsim import simulator as sim
from pyNN.serialization import import_from_sonata


def create_brain():
    """
    Initializes PyNN with the neural network that has to be simulated
    """

    return import_from_sonata('scaffold_sonata/circuit_config.json')


circuit = create_brain()
external_gloms = circuit.get_component('external_gloms').get_population('0')
scaffold = circuit.get_component('scaffold').get_population('0')

