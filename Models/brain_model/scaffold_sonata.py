# -*- coding: utf-8 -*-
"""
This file contains the setup of the neuronal network running the Husky experiment with neuronal image recognition
"""
# pragma: no cover

__author__ = 'Stefano Nardo'

from hbp_nrp_cle.brainsim import simulator as sim
from pyNN.serialization import import_from_sonata
import logging

logger = logging.getLogger(__name__)


def create_brain():
    """
    Initializes PyNN with the neuronal network that has to be simulated
    """
    network = import_from_sonata('scaffold_sonata/circuit_config.json', sim)

    return network


circuit = create_brain()

mossy = circuit.get_component('mossy').get_population('0')
dcn = circuit.get_component('dcn').get_population('0')
golgi = circuit.get_component('golgi').get_population('0')
dcn_interneuron = circuit.get_component('dcn_interneuron').get_population('0')
io = circuit.get_component('io').get_population('0')
basket = circuit.get_component('basket').get_population('0')
purkinje = circuit.get_component('purkinje').get_population('0')
stellate = circuit.get_component('stellate').get_population('0')
granule = circuit.get_component('granule').get_population('0')

