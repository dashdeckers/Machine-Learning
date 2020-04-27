import json
import os

from tensorflow.keras import layers

path = 'new_folder'
fn = 'experiment.json'
with open(os.path.join(path, fn), 'r') as file:
    exp = json.load(file)

test_layer = exp['encoder_layers'][0]


funcs = {
    'dense': layers.Dense,
    'flatten': layers.Flatten,
    'reshape': layers.Reshape,
    'conv2d': None,
}

# Convert layer configs into actual layers
for enc_dec in ['encoder_layers', 'decoder_layers']:
    for idx, config in enumerate(exp[enc_dec]):
        for fname in funcs.keys():
            if config['name'].startswith(fname):
                exp[enc_dec][idx] = funcs[fname].from_config(config)
